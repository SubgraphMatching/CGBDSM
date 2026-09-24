#include "index/cpu_pre_matcher.h"
#include <algorithm>

CPUPreMatcher::CPUPreMatcher(const CaLiG& calig, const CaLiGHelper* calig_helper,
                             const QueryGraph& query, const Plan& plan, uint32_t dv_count)
    : calig_(calig), calig_helper_(calig_helper), query_(query), plan_(plan),
      max_depth_(2), dv_count_(dv_count) {
    // 预先构建原始顶点ID到压缩索引的映射
    for (uint32_t vi = 0; vi < calig_.G.size(); vi++) {
        uint32_t compressed = calig_helper_->getCompressedVertexId(vi);
        if (compressed != UINT32_MAX) {
            vid2compressed_[vi] = compressed;
        }
    }
}

// 检查顶点u和数据顶点v是否与当前path兼容
bool CPUPreMatcher::checkAllCompatibility(uint32_t u, uint32_t v, const std::vector<uint32_t>& path) {
    // 只检查v是否与path中的已有顶点重复
    for (uint32_t pv : path) {
        if (pv == v) {
            return false;
        }
    }
    return true;
}

// 获取当前顶点u的所有候选数据顶点（利用LI结构，返回压缩索引）
std::vector<uint32_t> CPUPreMatcher::getCandidatesForVertex(uint32_t u, uint32_t depth, const std::vector<uint32_t>& path) {
    std::vector<uint32_t> candidates;

    // 从LI获取所有匹配u的顶点，转换为压缩索引
    for (uint32_t vi = 0; vi < calig_.G.size(); vi++) {
        auto li_it = calig_.G[vi].LI.find(u);
        if (li_it != calig_.G[vi].LI.end() && li_it->second) {
            // 使用预构建的映射
            auto it = vid2compressed_.find(vi);
            if (it == vid2compressed_.end()) continue;
            uint32_t compressed_vi = it->second;

            // 检查是否在path中已访问
            bool visited = false;
            for (uint32_t pv : path) {
                if (pv == compressed_vi) {
                    visited = true;
                    break;
                }
            }
            if (!visited) {
                candidates.push_back(compressed_vi);
            }
        }
    }

    return candidates;
}

// DFS递归匹配（只做顶点去重，让GPU来做完整的边兼容性检查）
void CPUPreMatcher::dfsMatch(std::vector<uint32_t>& path,
                             std::vector<bool>& visited,
                             uint32_t depth,
                             std::vector<CPUPartialMatch>& local_results) {
    if (depth == max_depth_) {
        local_results.push_back({path});
        return;
    }

    uint32_t u = plan_.orders_[0].vs_[depth];
    std::vector<uint32_t> candidates = getCandidatesForVertex(u, depth, path);

    for (uint32_t v : candidates) {
        if (visited[v]) continue;
        if (!checkAllCompatibility(u, v, path)) continue;

        visited[v] = true;
        path.push_back(v);
        dfsMatch(path, visited, depth + 1, local_results);
        path.pop_back();
        visited[v] = false;
    }
}

void CPUPreMatcher::cpuPreMatch(uint32_t k) {
    // 使用深度2以避免OOM，GPU会继续完成剩余匹配
    max_depth_ = 2;

    const uint32_t u0 = plan_.orders_[0].vs_[0];
    const uint32_t u1 = plan_.orders_[0].vs_[1];

    // OpenMP并行化
    #pragma omp parallel
    {
        std::vector<uint32_t> path;
        std::vector<bool> visited(dv_count_, false);  // 使用压缩索引空间大小
        std::vector<CPUPartialMatch> local_results;

        // 获取u0的候选
        std::vector<uint32_t> candidates_u0;
        for (uint32_t vi = 0; vi < calig_.G.size(); vi++) {
            auto li_it = calig_.G[vi].LI.find(u0);
            if (li_it != calig_.G[vi].LI.end() && li_it->second) {
                auto it = vid2compressed_.find(vi);
                if (it != vid2compressed_.end()) {
                    candidates_u0.push_back(it->second);
                }
            }
        }

        #pragma omp for nowait
        for (size_t idx = 0; idx < candidates_u0.size(); idx++) {
            uint32_t v0 = candidates_u0[idx];
            visited[v0] = true;
            path.push_back(v0);

            // 获取u1的候选
            std::vector<uint32_t> candidates_u1;
            for (uint32_t vi = 0; vi < calig_.G.size(); vi++) {
                auto li_it = calig_.G[vi].LI.find(u1);
                if (li_it != calig_.G[vi].LI.end() && li_it->second) {
                    auto it = vid2compressed_.find(vi);
                    if (it != vid2compressed_.end() && !visited[it->second]) {
                        candidates_u1.push_back(it->second);
                    }
                }
            }

            for (uint32_t v1 : candidates_u1) {
                visited[v1] = true;
                path.push_back(v1);

                // DFS继续匹配到深度k (depth=2, so matches vs_[2] if max_depth_ > 2)
                dfsMatch(path, visited, 2, local_results);

                path.pop_back();
                visited[v1] = false;
            }

            path.pop_back();
            visited[v0] = false;
        }

        // 合并到全局结果
        #pragma omp critical
        results_.insert(results_.end(), local_results.begin(), local_results.end());
    }
}