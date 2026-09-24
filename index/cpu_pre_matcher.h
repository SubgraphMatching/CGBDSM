#ifndef CPU_PRE_MATCHER_H
#define CPU_PRE_MATCHER_H

#include <vector>
#include <cstdint>
#include <unordered_map>
#include "index/calig.h"
#include "graph/graph.h"
#include "graph/plan.h"
#include "index/calig_helper.h"

struct CPUPartialMatch {
    std::vector<uint32_t> vertices;  // 匹配的k个数据顶点（压缩索引）
};

class CPUPreMatcher {
private:
    const CaLiG& calig_;
    const CaLiGHelper* calig_helper_;
    const QueryGraph& query_;
    const Plan& plan_;
    std::vector<CPUPartialMatch> results_;
    uint32_t max_depth_;
    uint32_t dv_count_;
    std::unordered_map<uint32_t, uint32_t> vid2compressed_;  // 原始顶点ID到压缩索引的映射

public:
    CPUPreMatcher(const CaLiG& calig, const CaLiGHelper* calig_helper, const QueryGraph& query, const Plan& plan, uint32_t dv_count);

    // 执行深度优先的CPU预匹配
    // 返回匹配到深度=k的部分匹配结果
    void cpuPreMatch(uint32_t k);

    // 获取CPU预匹配结果（用于传递给GPU）
    const std::vector<CPUPartialMatch>& getResults() const { return results_; }
    uint32_t getMaxDepth() const { return max_depth_; }
    uint32_t getResultCount() const { return results_.size(); }

private:
    // 获取当前顶点u的候选（利用已有匹配信息加速）
    std::vector<uint32_t> getCandidatesForVertex(uint32_t u, uint32_t depth, const std::vector<uint32_t>& path);

    // 检查顶点u和数据顶点v是否与当前path兼容
    bool checkAllCompatibility(uint32_t u, uint32_t v, const std::vector<uint32_t>& path);

    // DFS递归匹配
    void dfsMatch(std::vector<uint32_t>& path,
                  std::vector<bool>& visited,
                  uint32_t depth,
                  std::vector<CPUPartialMatch>& local_results);
};

#endif // CPU_PRE_MATCHER_H