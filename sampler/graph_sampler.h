#ifndef SAMPLER_GRAPH_SAMPLER_H
#define SAMPLER_GRAPH_SAMPLER_H

#include "graph.h"
#include <vector>
#include <queue>
#include <stack>
#include <random>
#include <algorithm>
#include <fstream>
#include <set>
#include <tuple>
#include <unordered_map>

enum class QueryType { Dense, Sparse, Tree };

struct SampledSubgraph {
    std::vector<int> vertices;
    std::vector<int> labels;
    std::vector<std::tuple<int,int,int>> edges;
    std::vector<int> original_ids;  // 原始顶点 ID，用于去重
};

struct BFSResult {
    std::unordered_set<int> visited;
    std::vector<std::pair<int,int>> tree_edges;
};

class GraphSampler {
public:
    GraphSampler(const Graph& graph, uint64_t seed = 0)
        : graph_(graph), rng_(seed ? seed : std::random_device{}()) {
        valid_vertices_.reserve(graph_.size);
        for (int v = 0; v < graph_.size; v++) {
            if (graph_.vertices[v].label != -1)
                valid_vertices_.push_back(v);
        }
    }

    SampledSubgraph sample(int max_size, QueryType type) {
        if (valid_vertices_.empty()) return {};
        std::uniform_int_distribution<int> dist(0, (int)valid_vertices_.size() - 1);
        int start = valid_vertices_[dist(rng_)];
        auto bfs = bfsFromWithTreeEdges(start, max_size);
        return buildSampledSubgraph(bfs, type);
    }

    std::vector<SampledSubgraph> sampleMultiple(
            int max_size, int num_samples, QueryType type = QueryType::Dense) {
        std::vector<SampledSubgraph> results;
        std::unordered_set<int> used_starts;
        std::set<std::vector<int>> used_vertex_sets;

        auto candidates = valid_vertices_;
        std::shuffle(candidates.begin(), candidates.end(), rng_);

        int cand_idx = 0;
        int max_attempts = num_samples * 1000;
        int attempts = 0;

        while ((int)results.size() < num_samples && attempts < max_attempts) {
            attempts++;
            int start = -1;
            while (cand_idx < (int)candidates.size()) {
                int v = candidates[cand_idx++];
                if (used_starts.find(v) == used_starts.end()) {
                    start = v;
                    used_starts.insert(v);
                    break;
                }
            }
            if (start == -1) break;

            auto bfs = (type == QueryType::Tree)
                ? dfsFromWithTreeEdges(start, max_size)
                : bfsFromWithTreeEdges(start, max_size);
            if ((int)bfs.visited.size() < max_size) continue;

            // Tree 类型：验证深度，拒绝星型结构
            if (type == QueryType::Tree) {
                int min_depth = std::max(2, (max_size + 2) / 3);
                int depth = computeTreeDepth(start, bfs.tree_edges);
                if (depth < min_depth) continue;
            }

            // 用原始 ID 去重
            std::vector<int> key(bfs.visited.begin(), bfs.visited.end());
            std::sort(key.begin(), key.end());
            if (used_vertex_sets.count(key)) continue;
            used_vertex_sets.insert(key);

            auto sg = buildSampledSubgraph(bfs, type);
            if (sg.vertices.empty()) continue;  // Dense 不满足条件

            results.push_back(std::move(sg));
        }
        return results;
    }

    static void writeToFile(const SampledSubgraph& sg, const std::string& path) {
        std::ofstream out(path);
        for (int i = 0; i < (int)sg.vertices.size(); i++)
            out << "v " << sg.vertices[i] << " " << sg.labels[i] << "\n";
        for (auto& [v, u, lb] : sg.edges)
            out << "e " << v << " " << u << " " << lb << "\n";
        out.close();
    }

    static double avgDegree(const SampledSubgraph& sg) {
        if (sg.vertices.empty()) return 0.0;
        return 2.0 * sg.edges.size() / sg.vertices.size();
    }

    // 检查子图是否连通（BFS 从第一个顶点出发）
    static bool isConnected(const SampledSubgraph& sg) {
        if (sg.vertices.empty()) return true;
        int n = (int)sg.vertices.size();
        std::unordered_map<int, int> id_to_idx;
        for (int i = 0; i < n; i++) id_to_idx[sg.vertices[i]] = i;

        std::vector<bool> visited(n, false);
        std::queue<int> q;
        q.push(0);
        visited[0] = true;
        int count = 1;

        while (!q.empty()) {
            int idx = q.front(); q.pop();
            for (auto& [v, u, lb] : sg.edges) {
                int neighbor = -1;
                if (v == sg.vertices[idx]) neighbor = u;
                else if (u == sg.vertices[idx]) neighbor = v;
                if (neighbor == -1) continue;
                auto it = id_to_idx.find(neighbor);
                if (it != id_to_idx.end() && !visited[it->second]) {
                    visited[it->second] = true;
                    q.push(it->second);
                    count++;
                }
            }
        }
        return count == n;
    }

private:
    const Graph& graph_;
    std::mt19937 rng_;
    std::vector<int> valid_vertices_;

    BFSResult bfsFromWithTreeEdges(int start, int max_size) {
        BFSResult result;
        std::queue<int> q;
        q.push(start);
        result.visited.insert(start);

        while (!q.empty() && (int)result.visited.size() < max_size) {
            int v = q.front(); q.pop();
            auto& nei = graph_.vertices[v].nei;
            std::vector<int> neighbors(nei.begin(), nei.end());
            std::shuffle(neighbors.begin(), neighbors.end(), rng_);

            int remaining = max_size - (int)result.visited.size();
            int limit = std::min(remaining, (int)neighbors.size());
            for (int i = 0; i < limit; i++) {
                int u = neighbors[i];
                if (u < graph_.size && graph_.vertices[u].label != -1
                    && result.visited.find(u) == result.visited.end()) {
                    result.visited.insert(u);
                    result.tree_edges.emplace_back(v, u);
                    q.push(u);
                }
            }
        }
        return result;
    }

    // DFS 版本的树生长，产生更深的树（避免星型结构）
    BFSResult dfsFromWithTreeEdges(int start, int max_size) {
        BFSResult result;
        std::stack<int> s;
        s.push(start);
        result.visited.insert(start);

        while (!s.empty() && (int)result.visited.size() < max_size) {
            int v = s.top(); s.pop();
            auto& nei = graph_.vertices[v].nei;
            std::vector<int> neighbors(nei.begin(), nei.end());
            std::shuffle(neighbors.begin(), neighbors.end(), rng_);

            int remaining = max_size - (int)result.visited.size();
            int limit = std::min(remaining, (int)neighbors.size());
            for (int i = 0; i < limit; i++) {
                int u = neighbors[i];
                if (u < graph_.size && graph_.vertices[u].label != -1
                    && result.visited.find(u) == result.visited.end()) {
                    result.visited.insert(u);
                    result.tree_edges.emplace_back(v, u);
                    s.push(u);
                }
            }
        }
        return result;
    }

    // 计算树从 root 出发的最大深度
    static int computeTreeDepth(int root, const std::vector<std::pair<int,int>>& tree_edges) {
        std::unordered_map<int, std::vector<int>> adj;
        for (auto& [v, u] : tree_edges) {
            adj[v].push_back(u);
            adj[u].push_back(v);
        }
        std::unordered_set<int> visited;
        std::queue<std::pair<int,int>> q;
        q.push({root, 0});
        visited.insert(root);
        int max_depth = 0;
        while (!q.empty()) {
            auto [v, d] = q.front(); q.pop();
            max_depth = std::max(max_depth, d);
            for (int u : adj[v]) {
                if (visited.find(u) == visited.end()) {
                    visited.insert(u);
                    q.push({u, d + 1});
                }
            }
        }
        return max_depth;
    }

    SampledSubgraph buildSampledSubgraph(BFSResult& bfs, QueryType type) {
        if (type == QueryType::Tree) {
            return buildSubgraph(bfs.visited, bfs.tree_edges);
        }

        // 计算所有诱导边
        std::vector<std::pair<int,int>> all_edges;
        for (int v : bfs.visited) {
            for (int u : graph_.vertices[v].nei) {
                if (v < u && bfs.visited.count(u))
                    all_edges.emplace_back(v, u);
            }
        }

        if (type == QueryType::Dense) {
            double davg = 2.0 * all_edges.size() / bfs.visited.size();
            if (davg < 3.0) return {};
            return buildSubgraph(bfs.visited, all_edges);
        }

        // Sparse: 从 BFS 树边开始，随机添加诱导边，保持 davg < 3
        auto edges = bfs.tree_edges;
        int max_edges = (3 * (int)bfs.visited.size()) / 2 - 1;

        std::set<std::pair<int,int>> tree_set;
        for (auto& [v, u] : edges) tree_set.insert(std::minmax(v, u));

        std::vector<std::pair<int,int>> extra_edges;
        for (auto& [v, u] : all_edges) {
            if (!tree_set.count(std::minmax(v, u)))
                extra_edges.emplace_back(v, u);
        }
        std::shuffle(extra_edges.begin(), extra_edges.end(), rng_);

        for (auto& e : extra_edges) {
            if ((int)edges.size() >= max_edges) break;
            edges.push_back(e);
        }

        // 确保不是 Tree（至少添加了 1 条非树边）
        if (edges.size() == bfs.tree_edges.size()) return {};

        return buildSubgraph(bfs.visited, edges);
    }

    SampledSubgraph buildSubgraph(
            const std::unordered_set<int>& visited,
            const std::vector<std::pair<int,int>>& edges) {
        SampledSubgraph result;

        std::vector<int> sorted_ids(visited.begin(), visited.end());
        std::sort(sorted_ids.begin(), sorted_ids.end());
        std::unordered_map<int, int> old_to_new;
        for (int i = 0; i < (int)sorted_ids.size(); i++)
            old_to_new[sorted_ids[i]] = i;

        result.vertices.reserve(sorted_ids.size());
        result.labels.reserve(sorted_ids.size());
        result.original_ids = sorted_ids;
        for (int old_id : sorted_ids) {
            result.vertices.push_back(old_to_new[old_id]);
            result.labels.push_back(graph_.vertices[old_id].label);
        }
        result.edges.reserve(edges.size());
        for (auto& [v, u] : edges)
            result.edges.emplace_back(old_to_new[v], old_to_new[u], 0);

        return result;
    }
};

#endif
