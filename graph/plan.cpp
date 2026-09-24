#include <algorithm>
#include <array>
#include <bitset>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

#include "graph/plan.h"
#include "graph/graph.h"

Plan::Plan(const QueryGraph& query_graph)
: query_(query_graph)
, rebuild_flags_{}
, rebuild_v_flags_{}
, cartesian_product_info_{}
, indexing_orders_{}
, orders_{}
{}

void Plan::GenerateIndexingOrders_v2()
{
    // first build indexing orders
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        const auto& [u0, u1] = query_.qe_list_[i];
        BuildIndexingOrder(u0, u1, indexing_orders_[i], indexing_ext_[i]);
    }
}

void Plan::GenerateMatchingOrders_v2(uint32_t *cardinalities, float *avg_degrees)
{
#ifdef USE_MERGED_MATCHING
    // 1. generate global matching order
#ifdef USE_GSI_ORDER
    GenerateGlobalMatchingOrderGSI(cardinalities, avg_degrees);
#else
    GenerateGlobalMatchingOrder(cardinalities, avg_degrees);
#endif

    // 2. per-edge metadata
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        const auto& [u0, u1] = query_.qe_list_[i];

        // all edges share global order
        orders_[i] = global_order_;
        cartesian_product_info_[i] = global_cartesian_product_info_;

        // rebuild flags based on global starting edge
        GenerateRebuildFlagsWithOrder(u0, u1,
            {global_order_.vs_[0], global_order_.vs_[1]}, rebuild_flags_[i]);
        GenerateRebuildVFlagsWithOrder(u0, u1, i);
    }
#else
    std::vector<uint8_t> cur_order;
    std::bitset<MAX_VCOUNT> visited;

    // build matching order for each query edge based on the local index
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        const auto& [u0, u1] = query_.qe_list_[i];
        cur_order.clear();
        visited.reset();

        // 1. ******************** find the starting edge ********************
#ifdef FORCE_UPDATE_EDGE_START
        visited[u0] = true;
        visited[u1] = true;
        cur_order.push_back(u0);
        cur_order.push_back(u1);
#else
        std::vector<std::pair<uint8_t, uint8_t>> selected_edges, further_selected_edges;

        // choose the edges whose endpoints have a maximum sum degree
        auto max_sum_degree = 0u;
        for (auto i = 0u; i < query_.ecount_; i++)
        {
            auto sum_degree = query_.nbrs_[query_.qe_list_[i].first].size() + query_.nbrs_[query_.qe_list_[i].second].size();
            if (sum_degree > max_sum_degree)
            {
                max_sum_degree = sum_degree;
                selected_edges.clear();
                selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
            }
            else if (sum_degree == max_sum_degree)
            {
                selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
            }
        }
        // if there is a tie, choose the edges that form the greatest number of triangles
        if (selected_edges.size() > 1)
        {
            auto max_num_triangles = 0u;
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto num_triangles = 0u;
                for (auto nbr = 0u; nbr < query_.vcount_; nbr++)
                {
                    auto found_uu0 = false, found_uu1 = false;
                    for (const auto& [u, _]: query_.nbrs_[nbr])
                    {
                        if (u == uu0) found_uu0 = true;
                        else if (u == uu1) found_uu1 = true;
                    }
                    if (found_uu0 && found_uu1) num_triangles++;
                }
                if (num_triangles > max_num_triangles)
                {
                    max_num_triangles = num_triangles;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (num_triangles == max_num_triangles)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        // if one of the selected edge is the updated edge, choose it as the starting edge of the order
        if (selected_edges.size() > 1)
        {
            for (const auto& [uu0, uu1]: selected_edges)
            {
                if ((uu0 == u0 && uu1 == u1) || (uu0 == u1 && uu1 == u0))
                {
                    further_selected_edges.emplace_back(u0, u1);
                    std::swap(selected_edges, further_selected_edges);
                    further_selected_edges.clear();
                    break;
                }
            }
        }
        // if there is a tie, choose the edges with the least number of candidates
        if (selected_edges.size() > 1)
        {
            auto min_cardinality = UINT32_MAX;
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto cardinality = cardinalities[query_.eidx_[uu0 * query_.vcount_ + uu1]];
                if (cardinality < min_cardinality)
                {
                    min_cardinality = cardinality;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (cardinality == min_cardinality)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        // if there is a tie, choose the edges whose neighbor have the least number of candidates
        if (selected_edges.size() > 1)
        {
            auto min_sum_cardinality = std::numeric_limits<float>::max();
            for (const auto& [uu0, uu1]: selected_edges)
            {
                auto sum_cardinality = 0.0f;
                for (const auto& [nbr, _]: query_.nbrs_[uu0])
                {
                    if (nbr == uu1) continue;
                    sum_cardinality += avg_degrees[query_.eidx_[uu0 * query_.vcount_ + nbr]];
                }
                for (const auto& [nbr, _]: query_.nbrs_[uu1])
                {
                    if (nbr == uu0) continue;
                    sum_cardinality += avg_degrees[query_.eidx_[uu1 * query_.vcount_ + nbr]];
                }
                if (sum_cardinality < min_sum_cardinality)
                {
                    min_sum_cardinality = sum_cardinality;
                    further_selected_edges.clear();
                    further_selected_edges.emplace_back(uu0, uu1);
                }
                else if (sum_cardinality == min_sum_cardinality)
                {
                    further_selected_edges.emplace_back(uu0, uu1);
                }
            }
            std::swap(selected_edges, further_selected_edges);
            further_selected_edges.clear();
        }
        
        // set the starting two vertices of the matching order
        visited[selected_edges[0].first] = true;
        visited[selected_edges[0].second] = true;
        cur_order.push_back(selected_edges[0].first);
        cur_order.push_back(selected_edges[0].second);
#endif

        // 2. ******************** add other vertices to the order ********************
        for (auto i = 2u; i < query_.vcount_; i++)
        {
            std::vector<uint8_t> selected, further_selected;

            // 0. gather all extendable vertices
            {
                std::vector<std::pair<uint8_t, uint32_t>> selected_with_avg_degree;
                for (auto u = 0u; u < query_.vcount_; u++)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            if (selected_with_avg_degree.empty() || selected_with_avg_degree.back().first != u)
                            {
                                selected_with_avg_degree.emplace_back(u, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                            }
                            else
                            {
                                if (avg_degrees[query_.eidx_[bn * query_.vcount_ + u]] < selected_with_avg_degree.back().second)
                                {
                                    selected_with_avg_degree.back().second = avg_degrees[query_.eidx_[bn * query_.vcount_ + u]];
                                }
                            }
                        }
                    }
                }
                // sort
                std::sort(
                    selected_with_avg_degree.begin(),
                    selected_with_avg_degree.end(),
                    [](const auto& p1, const auto& p2){
                        return p1.second < p2.second;
                    }
                );
                // remove vertices with extreme large avg degrees
                auto new_end = selected_with_avg_degree.size();
                for (auto j = 0u; j < selected_with_avg_degree.size() - 1; j++)
                {
                    if (selected_with_avg_degree[j + 1].second >= selected_with_avg_degree[j].second * (2 << 9))
                    {
                        new_end = j + 1;
                        break;
                    }
                }
                for (auto j = 0u; j < new_end; j++)
                {
                    selected.push_back(selected_with_avg_degree[j].first);
                }
                std::sort(selected.begin(), selected.end());
            }

            // 1. select the vertices with the maximum number of backward neighbors
            if (selected.size() > 1)
            {
                auto max_num_bn = 0u;
                for (const auto& u: selected)
                {
                    if (visited[u]) continue;
                    // count the number of backward neighbors
                    auto cur_num_bns = 0u;
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn]) cur_num_bns += 1;
                    }
                    if (cur_num_bns > max_num_bn)
                    {
                        max_num_bn = cur_num_bns;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_bns == max_num_bn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 2. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u
            if (selected.size() > 1)
            {
                auto max_num_v = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_v = 0u;
                    std::vector<bool> temp_visited(query_.vcount_, false);
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn] && !temp_visited[bn])
                                {
                                    temp_visited[bn] = true;
                                    cur_num_v++;
                                }
                            }
                        }
                    }
                    if (cur_num_v > max_num_v)
                    {
                        max_num_v = cur_num_v;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_v == max_num_v)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }

            // 3. if there is a tie, compute the number of vertices in the matching order that has at least one vertex
            // not in the matching order and connected with u
            if (selected.size() > 1)
            {
                auto max_num_fn = 0u;
                for (const auto& u: selected)
                {
                    auto cur_num_fn = 0u;
                    for (const auto& [fn, _]: query_.nbrs_[u])
                    {
                        if (!visited[fn])
                        {
                            auto no_visited_nbr_of_fn = true;
                            for (const auto& [bn, _]: query_.nbrs_[fn])
                            {
                                if (visited[bn])
                                {
                                    no_visited_nbr_of_fn = false;
                                    break;
                                }
                            }
                            if (no_visited_nbr_of_fn)
                            {
                                cur_num_fn++;
                            }
                        }
                    }
                    if (cur_num_fn > max_num_fn)
                    {
                        max_num_fn = cur_num_fn;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_fn == max_num_fn)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // 4. if there is a tie, select a vertex with the minimum number of candiates on extension,
            // where the number of candidates is estimated by the min property
            if (selected.size() > 1)
            {
                auto min_num_candidates = std::numeric_limits<float>::max();
                for (const auto& u: selected)
                {
                    auto cur_num_candidates = std::numeric_limits<float>::max();
                    for (const auto& [bn, _]: query_.nbrs_[u])
                    {
                        if (visited[bn])
                        {
                            cur_num_candidates = std::min(cur_num_candidates, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                        }
                    }
                    if (cur_num_candidates < min_num_candidates)
                    {
                        min_num_candidates = cur_num_candidates;
                        further_selected.clear();
                        further_selected.push_back(u);
                    }
                    else if (cur_num_candidates == min_num_candidates)
                    {
                        further_selected.push_back(u);
                    }
                }
                std::swap(further_selected, selected);
                further_selected.clear();
            }
            // insert the first selected vertex to the matching order
            cur_order.push_back(selected[0]);
            visited[selected[0]] = true;
        }
        // generate rebuild_flags_
        GenerateRebuildFlagsWithOrder(u0, u1, {cur_order[0], cur_order[1]}, rebuild_flags_[i]);
        GenerateRebuildVFlagsWithOrder(u0, u1, i);
        GenerateCartesianProductInfo(cur_order, i);

        // generate meta for the matching order
        auto cum_offs = 0u;
        orders_[i].bni_offs_[0] = cum_offs;
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            const auto& u = cur_order[j];
            orders_[i].vs_[j] = u;
            for (auto k = j - 1; k < query_.vcount_; k--)
            {
                const auto& uu = orders_[i].vs_[k];
                auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
                if (it != query_.nbrs_[u].end() && it->first == uu)
                {
                    orders_[i].bni_[cum_offs++] = k;
                }
            }
            orders_[i].bni_offs_[j + 1] = cum_offs;
            std::sort(
                orders_[i].bni_ + orders_[i].bni_offs_[j],
                orders_[i].bni_ + orders_[i].bni_offs_[j + 1],
                [this, i, u, avg_degrees](const auto& bni1, const auto& bni2){
                    return avg_degrees[this->query_.eidx_[this->orders_[i].vs_[bni1] * this->query_.vcount_ + u]]
                    < avg_degrees[this->query_.eidx_[this->orders_[i].vs_[bni2] * this->query_.vcount_ + u]];
                }
            );
        }
    }
#endif // USE_MERGED_MATCHING
}

#ifdef USE_MERGED_MATCHING

std::pair<uint8_t, uint8_t> Plan::SelectStartingEdge(uint32_t *cardinalities, float *avg_degrees)
{
    std::vector<std::pair<uint8_t, uint8_t>> selected_edges, further_selected_edges;

    // Level 1: choose the edges whose endpoints have a maximum sum degree
    auto max_sum_degree = 0u;
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        auto sum_degree = query_.nbrs_[query_.qe_list_[i].first].size() + query_.nbrs_[query_.qe_list_[i].second].size();
        if (sum_degree > max_sum_degree)
        {
            max_sum_degree = sum_degree;
            selected_edges.clear();
            selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
        }
        else if (sum_degree == max_sum_degree)
        {
            selected_edges.emplace_back(query_.qe_list_[i].first, query_.qe_list_[i].second);
        }
    }
    // Level 2: if there is a tie, choose the edges that form the greatest number of triangles
    if (selected_edges.size() > 1)
    {
        auto max_num_triangles = 0u;
        for (const auto& [uu0, uu1]: selected_edges)
        {
            auto num_triangles = 0u;
            for (auto nbr = 0u; nbr < query_.vcount_; nbr++)
            {
                auto found_uu0 = false, found_uu1 = false;
                for (const auto& [u, _]: query_.nbrs_[nbr])
                {
                    if (u == uu0) found_uu0 = true;
                    else if (u == uu1) found_uu1 = true;
                }
                if (found_uu0 && found_uu1) num_triangles++;
            }
            if (num_triangles > max_num_triangles)
            {
                max_num_triangles = num_triangles;
                further_selected_edges.clear();
                further_selected_edges.emplace_back(uu0, uu1);
            }
            else if (num_triangles == max_num_triangles)
            {
                further_selected_edges.emplace_back(uu0, uu1);
            }
        }
        std::swap(selected_edges, further_selected_edges);
        further_selected_edges.clear();
    }
    // Level 3: if there is a tie, choose the edges with the least number of candidates
    if (selected_edges.size() > 1)
    {
        auto min_cardinality = UINT32_MAX;
        for (const auto& [uu0, uu1]: selected_edges)
        {
            auto cardinality = cardinalities[query_.eidx_[uu0 * query_.vcount_ + uu1]];
            if (cardinality < min_cardinality)
            {
                min_cardinality = cardinality;
                further_selected_edges.clear();
                further_selected_edges.emplace_back(uu0, uu1);
            }
            else if (cardinality == min_cardinality)
            {
                further_selected_edges.emplace_back(uu0, uu1);
            }
        }
        std::swap(selected_edges, further_selected_edges);
        further_selected_edges.clear();
    }
    // Level 4: if there is a tie, choose the edges whose neighbor have the least number of candidates
    if (selected_edges.size() > 1)
    {
        auto min_sum_cardinality = std::numeric_limits<float>::max();
        for (const auto& [uu0, uu1]: selected_edges)
        {
            auto sum_cardinality = 0.0f;
            for (const auto& [nbr, _]: query_.nbrs_[uu0])
            {
                if (nbr == uu1) continue;
                sum_cardinality += avg_degrees[query_.eidx_[uu0 * query_.vcount_ + nbr]];
            }
            for (const auto& [nbr, _]: query_.nbrs_[uu1])
            {
                if (nbr == uu0) continue;
                sum_cardinality += avg_degrees[query_.eidx_[uu1 * query_.vcount_ + nbr]];
            }
            if (sum_cardinality < min_sum_cardinality)
            {
                min_sum_cardinality = sum_cardinality;
                further_selected_edges.clear();
                further_selected_edges.emplace_back(uu0, uu1);
            }
            else if (sum_cardinality == min_sum_cardinality)
            {
                further_selected_edges.emplace_back(uu0, uu1);
            }
        }
        std::swap(selected_edges, further_selected_edges);
        further_selected_edges.clear();
    }

    return {selected_edges[0].first, selected_edges[0].second};
}

void Plan::GenerateGlobalMatchingOrder(uint32_t *cardinalities, float *avg_degrees)
{
    std::vector<uint8_t> cur_order;
    std::bitset<MAX_VCOUNT> visited;
    cur_order.clear();
    visited.reset();

    // 1. ******************** find the starting edge (global, no "prefer update edge") ********************
    auto [u_first, u_second] = SelectStartingEdge(cardinalities, avg_degrees);

    // set the starting two vertices
    visited[u_first] = true;
    visited[u_second] = true;
    cur_order.push_back(u_first);
    cur_order.push_back(u_second);

    // 2. ******************** add other vertices (same 4-level criterion) ********************
    for (auto i = 2u; i < query_.vcount_; i++)
    {
        std::vector<uint8_t> selected, further_selected;

        // 0. gather all extendable vertices
        {
            std::vector<std::pair<uint8_t, uint32_t>> selected_with_avg_degree;
            for (auto u = 0u; u < query_.vcount_; u++)
            {
                if (visited[u]) continue;
                for (const auto& [bn, _]: query_.nbrs_[u])
                {
                    if (visited[bn])
                    {
                        if (selected_with_avg_degree.empty() || selected_with_avg_degree.back().first != u)
                        {
                            selected_with_avg_degree.emplace_back(u, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                        }
                        else
                        {
                            if (avg_degrees[query_.eidx_[bn * query_.vcount_ + u]] < selected_with_avg_degree.back().second)
                            {
                                selected_with_avg_degree.back().second = avg_degrees[query_.eidx_[bn * query_.vcount_ + u]];
                            }
                        }
                    }
                }
            }
            std::sort(selected_with_avg_degree.begin(), selected_with_avg_degree.end(),
                [](const auto& p1, const auto& p2){ return p1.second < p2.second; });
            auto new_end = selected_with_avg_degree.size();
            for (auto j = 0u; j < selected_with_avg_degree.size() - 1; j++)
            {
                if (selected_with_avg_degree[j + 1].second >= selected_with_avg_degree[j].second * (2 << 9))
                {
                    new_end = j + 1;
                    break;
                }
            }
            for (auto j = 0u; j < new_end; j++)
                selected.push_back(selected_with_avg_degree[j].first);
            std::sort(selected.begin(), selected.end());
        }

        // 1. max backward neighbors
        if (selected.size() > 1)
        {
            auto max_num_bn = 0u;
            for (const auto& u: selected)
            {
                if (visited[u]) continue;
                auto cur_num_bns = 0u;
                for (const auto& [bn, _]: query_.nbrs_[u])
                    if (visited[bn]) cur_num_bns += 1;
                if (cur_num_bns > max_num_bn)
                {
                    max_num_bn = cur_num_bns;
                    further_selected.clear();
                    further_selected.push_back(u);
                }
                else if (cur_num_bns == max_num_bn)
                    further_selected.push_back(u);
            }
            std::swap(further_selected, selected);
            further_selected.clear();
        }

        // 2. max visited neighbors of forward neighbors
        if (selected.size() > 1)
        {
            auto max_num_v = 0u;
            for (const auto& u: selected)
            {
                auto cur_num_v = 0u;
                std::vector<bool> temp_visited(query_.vcount_, false);
                for (const auto& [fn, _]: query_.nbrs_[u])
                {
                    if (!visited[fn])
                    {
                        for (const auto& [bn, _]: query_.nbrs_[fn])
                        {
                            if (visited[bn] && !temp_visited[bn])
                            {
                                temp_visited[bn] = true;
                                cur_num_v++;
                            }
                        }
                    }
                }
                if (cur_num_v > max_num_v)
                {
                    max_num_v = cur_num_v;
                    further_selected.clear();
                    further_selected.push_back(u);
                }
                else if (cur_num_v == max_num_v)
                    further_selected.push_back(u);
            }
            std::swap(further_selected, selected);
            further_selected.clear();
        }

        // 3. max dangling forward neighbors
        if (selected.size() > 1)
        {
            auto max_num_fn = 0u;
            for (const auto& u: selected)
            {
                auto cur_num_fn = 0u;
                for (const auto& [fn, _]: query_.nbrs_[u])
                {
                    if (!visited[fn])
                    {
                        auto no_visited_nbr_of_fn = true;
                        for (const auto& [bn, _]: query_.nbrs_[fn])
                        {
                            if (visited[bn]) { no_visited_nbr_of_fn = false; break; }
                        }
                        if (no_visited_nbr_of_fn) cur_num_fn++;
                    }
                }
                if (cur_num_fn > max_num_fn)
                {
                    max_num_fn = cur_num_fn;
                    further_selected.clear();
                    further_selected.push_back(u);
                }
                else if (cur_num_fn == max_num_fn)
                    further_selected.push_back(u);
            }
            std::swap(further_selected, selected);
            further_selected.clear();
        }

        // 4. min estimated candidates
        if (selected.size() > 1)
        {
            auto min_num_candidates = std::numeric_limits<float>::max();
            for (const auto& u: selected)
            {
                auto cur_num_candidates = std::numeric_limits<float>::max();
                for (const auto& [bn, _]: query_.nbrs_[u])
                {
                    if (visited[bn])
                        cur_num_candidates = std::min(cur_num_candidates, avg_degrees[query_.eidx_[bn * query_.vcount_ + u]]);
                }
                if (cur_num_candidates < min_num_candidates)
                {
                    min_num_candidates = cur_num_candidates;
                    further_selected.clear();
                    further_selected.push_back(u);
                }
                else if (cur_num_candidates == min_num_candidates)
                    further_selected.push_back(u);
            }
            std::swap(further_selected, selected);
            further_selected.clear();
        }

        cur_order.push_back(selected[0]);
        visited[selected[0]] = true;
    }

    // generate cartesian product info
    GenerateCartesianProductInfoGlobal(cur_order, global_cartesian_product_info_);

    // generate meta for the global matching order
    auto cum_offs = 0u;
    global_order_.bni_offs_[0] = cum_offs;
    for (auto j = 0u; j < query_.vcount_; j++)
    {
        const auto& u = cur_order[j];
        global_order_.vs_[j] = u;
        for (auto k = j - 1; k < query_.vcount_; k--)
        {
            const auto& uu = global_order_.vs_[k];
            auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
            if (it != query_.nbrs_[u].end() && it->first == uu)
            {
                global_order_.bni_[cum_offs++] = k;
            }
        }
        global_order_.bni_offs_[j + 1] = cum_offs;
        // sort backward neighbors by avg_degrees (ascending = most selective first)
        std::sort(
            global_order_.bni_ + global_order_.bni_offs_[j],
            global_order_.bni_ + global_order_.bni_offs_[j + 1],
            [this, u, avg_degrees](const auto& bni1, const auto& bni2){
                return avg_degrees[this->query_.eidx_[this->global_order_.vs_[bni1] * this->query_.vcount_ + u]]
                     < avg_degrees[this->query_.eidx_[this->global_order_.vs_[bni2] * this->query_.vcount_ + u]];
            }
        );
    }
}

void Plan::GenerateGlobalMatchingOrderGSI(uint32_t *cardinalities, float *avg_degrees)
{
    std::vector<uint8_t> cur_order;
    std::bitset<MAX_VCOUNT> visited;
    cur_order.clear();
    visited.reset();

    constexpr float epsilon = 1e-6f;

    // Step 1: Estimate |C(u)| for each query vertex
    // est_C[u] = min over neighbors v of (cardinalities[eidx(u,v)] / max(avg_degrees[eidx(u,v)], epsilon))
    // This estimates the number of distinct data vertices with candidates for u
    std::array<float, MAX_VCOUNT> est_C;
    est_C.fill(std::numeric_limits<float>::max());
    for (auto u = 0u; u < query_.vcount_; u++)
    {
        for (const auto& [v, _]: query_.nbrs_[u])
        {
            auto eidx = query_.eidx_[u * query_.vcount_ + v];
            float est = (float)cardinalities[eidx] / std::max(avg_degrees[eidx], epsilon);
            est_C[u] = std::min(est_C[u], est);
        }
    }

    // Step 2: Initialize scores = est_C[u] / deg(u)
    // Fewer candidates + higher degree -> lower score -> prefer first
    std::array<float, MAX_VCOUNT> score;
    for (auto u = 0u; u < query_.vcount_; u++)
    {
        score[u] = est_C[u] / std::max((float)query_.nbrs_[u].size(), epsilon);
    }

    // Step 3: Select the first vertex (argmin score), then immediately update scores
    // GSI Algorithm 2: select one vertex per iteration, update scores after each
    uint8_t u_first = 0;
    float min_score = score[0];
    for (auto u = 1u; u < query_.vcount_; u++)
    {
        if (score[u] < min_score)
        {
            min_score = score[u];
            u_first = u;
        }
    }
    visited[u_first] = true;
    cur_order.push_back(u_first);

    // Update scores of u_first's neighbors
    for (const auto& [v, _]: query_.nbrs_[u_first])
    {
        if (!visited[v])
            score[v] *= 0.9f;
    }

    // Step 4: Select the second vertex using UPDATED scores (neighbors of u_first)
    uint8_t u_second = query_.nbrs_[u_first][0].first;
    min_score = score[u_second];
    for (const auto& [v, _]: query_.nbrs_[u_first])
    {
        if (score[v] < min_score)
        {
            min_score = score[v];
            u_second = v;
        }
    }
    visited[u_second] = true;
    cur_order.push_back(u_second);

    // Update scores of u_second's unvisited neighbors
    for (const auto& [v, _]: query_.nbrs_[u_second])
    {
        if (!visited[v])
            score[v] *= 0.9f;
    }

    // Step 5: Greedy extension (i = 2 to vcount_-1)
    // Primary: argmin score (GSI edge sparsity awareness)
    // Secondary: max backward neighbors (structural pruning power)
    for (auto i = 2u; i < query_.vcount_; i++)
    {
        // Among unvisited vertices connected to the visited set
        float best_score = std::numeric_limits<float>::max();
        uint8_t best_u = 0;
        auto best_num_bn = 0u;
        for (auto u = 0u; u < query_.vcount_; u++)
        {
            if (visited[u]) continue;
            // Must be connected to the visited set
            auto cur_num_bn = 0u;
            bool connected = false;
            for (const auto& [v, _]: query_.nbrs_[u])
            {
                if (visited[v]) { connected = true; cur_num_bn++; }
            }
            if (!connected) continue;

            // Primary: lower score preferred
            // Secondary (tie-breaker): more backward neighbors preferred
            if (score[u] < best_score
                || (score[u] == best_score && cur_num_bn > best_num_bn))
            {
                best_score = score[u];
                best_u = u;
                best_num_bn = cur_num_bn;
            }
        }

        visited[best_u] = true;
        cur_order.push_back(best_u);

        // Update scores of unvisited neighbors
        for (const auto& [v, _]: query_.nbrs_[best_u])
        {
            if (!visited[v])
            {
                score[v] *= 0.9f;  // GSI reference: decay factor, not avg_degrees (would cause score explosion)
            }
        }
    }

    // Step 7: Post-processing (identical to RI)
    GenerateCartesianProductInfoGlobal(cur_order, global_cartesian_product_info_);

    auto cum_offs = 0u;
    global_order_.bni_offs_[0] = cum_offs;
    for (auto j = 0u; j < query_.vcount_; j++)
    {
        const auto& u = cur_order[j];
        global_order_.vs_[j] = u;
        for (auto k = j - 1; k < query_.vcount_; k--)
        {
            const auto& uu = global_order_.vs_[k];
            auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(),
                std::make_pair(static_cast<uint32_t>(uu), 0u));
            if (it != query_.nbrs_[u].end() && it->first == uu)
            {
                global_order_.bni_[cum_offs++] = k;
            }
        }
        global_order_.bni_offs_[j + 1] = cum_offs;
        // sort backward neighbors by avg_degrees (ascending = most selective first)
        std::sort(
            global_order_.bni_ + global_order_.bni_offs_[j],
            global_order_.bni_ + global_order_.bni_offs_[j + 1],
            [this, u, avg_degrees](const auto& bni1, const auto& bni2){
                return avg_degrees[this->query_.eidx_[this->global_order_.vs_[bni1] * this->query_.vcount_ + u]]
                     < avg_degrees[this->query_.eidx_[this->global_order_.vs_[bni2] * this->query_.vcount_ + u]];
            }
        );
    }
}

void Plan::GenerateCartesianProductInfoGlobal(
    std::vector<uint8_t>& cur_order,
    std::array<CartesianProductType, MAX_VCOUNT>& cp_info
) {
    CartesianProductType type;
    uint8_t vertex = cur_order[query_.vcount_ - 1];
    if (query_.nbrs_[vertex].size() == 1)
    {
        cp_info[query_.vcount_ - 1] = CartesianProductType::TreeSingle;
        type = CartesianProductType::TreeCartesianProduct;
    }
    else
    {
        cp_info[query_.vcount_ - 1] = CartesianProductType::NonTreeSingle;
        type = CartesianProductType::NonTreeCartesianProduct;
    }
    for (auto i = 1u; i < query_.vcount_ - 1; i++)
    {
        if (type == CartesianProductType::None)
        {
            cp_info[query_.vcount_ - 1 - i] = CartesianProductType::None;
        }
        else
        {
            vertex = cur_order[query_.vcount_ - 1 - i];
            bool valid = true;
            for (auto j = 0u; j < i; j++)
            {
                uint8_t v_other = cur_order[query_.vcount_ - 1 - j];
                auto it = std::lower_bound(query_.nbrs_[vertex].begin(), query_.nbrs_[vertex].end(), make_pair<uint32_t, uint32_t>(v_other, 0));
                if (it != query_.nbrs_[vertex].end() && it->first == v_other)
                {
                    valid = false;
                    break;
                }
            }
            if (valid)
            {
                if (query_.nbrs_[vertex].size() == 1 && type == CartesianProductType::TreeCartesianProduct)
                    cp_info[query_.vcount_ - 1 - i] = CartesianProductType::TreeCartesianProduct;
                else
                {
                    cp_info[query_.vcount_ - 1 - i] = CartesianProductType::NonTreeCartesianProduct;
                    type = CartesianProductType::NonTreeCartesianProduct;
                }
            }
            else
            {
                cp_info[query_.vcount_ - 1 - i] = CartesianProductType::None;
                type = CartesianProductType::None;
            }
        }
    }
}

#endif // USE_MERGED_MATCHING

void Plan::PrintOrders()
{
    std::cout << "Orders:\n";
    for (auto i = 0u; i < query_.ecount_; i++)
    {
        const auto& [u0, u1] = query_.qe_list_[i];
        std::cout << '[' << u0 << ',' << u1 << "]:\n    indexing order:";
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            std::cout << static_cast<uint32_t>(indexing_orders_[i].vs_[j]);
            std::cout << (rebuild_v_flags_[i][indexing_orders_[i].vs_[j]] ? "-build-(" : "-(");
            for (auto k = indexing_orders_[i].bni_offs_[j]; k < indexing_orders_[i].bni_offs_[j + 1]; k++)
            {
                std::cout << static_cast<uint32_t>(indexing_orders_[i].vs_[indexing_orders_[i].bni_[k]]) << '@'
                    <<static_cast<uint32_t>(indexing_orders_[i].bni_[k]);
                if (k != indexing_orders_[i].bni_offs_[j + 1] - 1)
                {
                    std::cout << ' ';
                }
            }
            std::cout << ") ";
        }
        std::cout << "\n    order: ";
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            std::cout << static_cast<uint32_t>(orders_[i].vs_[j]) << "-(";
            for (auto k = orders_[i].bni_offs_[j]; k < orders_[i].bni_offs_[j + 1]; k++)
            {
                std::cout << static_cast<uint32_t>(orders_[i].vs_[orders_[i].bni_[k]]) << '@'
                    << static_cast<uint32_t>(orders_[i].bni_[k]);
                if (k != orders_[i].bni_offs_[j + 1] - 1)
                {
                    std::cout << ' ';
                }
            }
            std::cout << ") ";
        }
        std::cout << "\n    cartesian product info: ";
        for (auto j = 0u; j < query_.vcount_; j++)
        {
            std::cout << static_cast<uint32_t>(orders_[i].vs_[j]);
            if (j == 0u)
            {
                std::cout << ' ';
            }
            else
            {
                std::cout << "-(" << static_cast<uint32_t>(cartesian_product_info_[i][j]) << ") ";
            }
        }
        std::cout << "\n";
    }
}

void Plan::GroupDenseVertices(
    std::vector<nd_tree_node>& k34_tree,
    std::vector<uint8_t>& vrole
) {
    std::vector<std::vector<uint32_t>> dense_vertices;
    // 1. get exclusive nucleus
    for (auto node: k34_tree)
    {
        auto merge_idx = UINT32_MAX;

        for (auto i = 0u; i < dense_vertices.size(); i++)
        {
            auto& nbrs = dense_vertices[i];
            std::vector<uint32_t> result(std::min(node.vertices_.size(), nbrs.size()));
            if (std::set_intersection(
                node.vertices_.begin(), node.vertices_.end(),
                nbrs.begin(), nbrs.end(),
                result.begin()) != result.begin()
            ) {
                // the new node have some vertex in common with dense_vertices[i]
                merge_idx = i;
                break;
            }
        }
        if (merge_idx == UINT32_MAX)
        {
            // create a new group
            merge_idx = dense_vertices.size();
            dense_vertices.emplace_back();
        }

        // merge all the vertices in node to an existing group
        std::vector<uint32_t> new_core(dense_vertices[merge_idx].size() + node.vertices_.size());
        auto it = std::set_union(
            dense_vertices[merge_idx].begin(), dense_vertices[merge_idx].end(),
            node.vertices_.begin(), node.vertices_.end(),
            new_core.begin()
        );
        new_core.resize(it - new_core.begin());
        std::swap(new_core, dense_vertices[merge_idx]);

    }
    // 2. sort the nucleus and number the vertices in each nucleus
    std::sort(
        dense_vertices.begin(), dense_vertices.end(),
        [](const auto& u1, const auto& v2){
            return u1.size() < v2.size();
        }
    );
    for (uint32_t i = 0u; i < dense_vertices.size(); i++)
    {
        for (auto v: dense_vertices[i])
        {
            vrole[v] = i + 2;
        }
    }
}

void Plan::AddVertices(
    std::vector<uint8_t>& order,
    std::bitset<MAX_VCOUNT>& visited,
    const std::vector<uint8_t>& vrole,
    const uint8_t group_id
) {
    // 1. find all initial extendable vertices
    std::bitset<MAX_VCOUNT> extendable;
    auto extendable_count = 0u;
    std::vector<uint8_t> extendable_score(query_.vcount_, 0);
    for (auto v = 0u; v < query_.vcount_; v++)
    {
        if (visited[v] || vrole[v] != group_id) continue;
        // check if v is connected with one vertex in the current order
        for (const auto& [nbr, label]: query_.nbrs_[v])
        {
            if (!visited[nbr]) continue;
            if (!extendable[v])
            {
                extendable_count += 1;
                extendable[v] = true;
            }
            extendable_score[v] += 1;
        }
    }

    // 2. move an extendable vertex to the matching order and update the extendable vertices
    while (extendable_count != 0)
    {
        auto selected_v = 0u, selected_v_score = 0u;
        for (auto i = 0u; i < query_.vcount_; i++)
        {
            if (extendable[i] && extendable_score[i] > selected_v_score)
            {
                selected_v_score = extendable_score[i];
                selected_v = i;
            }
        }
        order.push_back(selected_v);
        visited[selected_v] = true;
        extendable[selected_v] = false;
        extendable_count --;
        for (const auto& [nbr, label]: query_.nbrs_[selected_v])
        {
            if (visited[nbr] || vrole[nbr] != group_id) continue;
            if (!extendable[nbr])
            {
                extendable_count += 1;
                extendable[nbr] = true;
            }
            extendable_score[nbr] += 1;
        }
    }
}

void Plan::AddVertices(
    std::vector<uint8_t>& order,
    std::bitset<MAX_VCOUNT>& visited,
    const std::vector<uint8_t>& vrole
) {
    // 1. find all initial extendable vertices
    std::bitset<MAX_VCOUNT> extendable;
    auto extendable_count = 0u;
    std::vector<std::pair<uint8_t, uint8_t>> extendable_score(query_.vcount_, {0,0});
    for (auto v = 0u; v < query_.vcount_; v++)
    {
        // check if v is connected with on vertex in the current order
        if (visited[v]) continue;
        for (const auto& [nbr, label]: query_.nbrs_[v])
        {
            if (!visited[nbr]) continue;
            if (!extendable[v])
            {
                extendable_count += 1;
                extendable[v] = true;
                extendable_score[v].first = vrole[v] > 0 ? 1 : 0;
            }
            extendable_score[v].second += 1;
        }
    }

    // 2. move an extendable vertex to the matching order and update the extendable vertices
    while (extendable_count != 0)
    {
        uint8_t selected_v;
        std::pair<uint8_t, uint8_t> selected_v_score {0u, 0u};
        for (uint32_t i = 0; i < query_.vcount_; i++)
        {
            if (extendable[i] && extendable_score[i] > selected_v_score)
            {
                selected_v_score = extendable_score[i];
                selected_v = i;
            }
        }
        order.push_back(selected_v);
        visited[selected_v] = true;
        extendable[selected_v] = false;
        extendable_count --;
        for (const auto& [nbr, label]: query_.nbrs_[selected_v])
        {
            if (visited[nbr]) continue;
            if (!extendable[nbr])
            {
                extendable_count += 1;
                extendable[nbr] = true;
                extendable_score[nbr].first = vrole[nbr] > 0 ? 1 : 0;
            }
            extendable_score[nbr].second += 1;
        }
    }
}

void Plan::BuildIndexingOrder(
    const uint8_t u0, const uint8_t u1,
    OrderPerEdge& indexing_orders_,
    IndexingOrderExt& ext
) {
    // get the bfs order of the query vertices
    std::vector<uint8_t> cur_order {u0, u1};

    std::vector<uint8_t> pre_level {u0, u1};
    std::vector<uint8_t> cur_level;
    std::bitset<MAX_VCOUNT> visited;
    visited[u0] = true;
    visited[u1] = true;

    while (!pre_level.empty())
    {
        for (const auto& pre_v: pre_level)
        {
            for (const auto& [nbr, label]: query_.nbrs_[pre_v])
            {
                if (!visited[nbr])
                {
                    cur_level.push_back(nbr);
                    visited[nbr] = true;
                    cur_order.push_back(nbr);
                }
            }
        }
        pre_level.clear();
        std::swap(pre_level, cur_level);
    }

    // fill in the indexing_order
    indexing_orders_.vs_[0] = cur_order[0];
    indexing_orders_.bni_offs_[0] = 0u;
    indexing_orders_.bni_[0] = 1u;
    auto cum_offs = 1u;
    indexing_orders_.bni_offs_[1] = 1u;
    for (auto j = 1u; j < query_.vcount_; j++)
    {
        const auto& u = cur_order[j];
        indexing_orders_.vs_[j] = u;
        for (auto k = 0u; k < j; k++)
        {
            const auto& uu = indexing_orders_.vs_[k];
            auto it = std::lower_bound(query_.nbrs_[u].begin(), query_.nbrs_[u].end(), std::make_pair(static_cast<uint32_t>(uu), 0u));
            if (it != query_.nbrs_[u].end() && it->first == uu)
            {
                indexing_orders_.bni_[cum_offs++] = k;
            }
        }
        indexing_orders_.bni_offs_[j + 1] = cum_offs;
    }

    // Compute triangle pair metadata
    uint8_t tri_count = 0;
    for (uint8_t d = 2; d < query_.vcount_; d++) {
        ext.tri_offs_[d] = tri_count;
        for (uint8_t off_i = indexing_orders_.bni_offs_[d];
             off_i < indexing_orders_.bni_offs_[d + 1]; off_i++) {
            for (uint8_t off_j = off_i + 1;
                 off_j < indexing_orders_.bni_offs_[d + 1]; off_j++) {
                uint8_t u_i = indexing_orders_.vs_[indexing_orders_.bni_[off_i]];
                uint8_t u_j = indexing_orders_.vs_[indexing_orders_.bni_[off_j]];
                uint8_t eidx = query_.eidx_[u_i * query_.vcount_ + u_j];
                if (eidx != UINT8_MAX && tri_count < MAX_TRI_PAIRS) {
                    ext.tri_pairs_[tri_count++] = {off_i, off_j, eidx};
                }
            }
        }
    }
    ext.tri_offs_[query_.vcount_] = tri_count;
    for (uint8_t d = 0; d < 2; d++) ext.tri_offs_[d] = 0;

    // Compute forward neighbor metadata
    // Build inverse position map: pos[u] = depth where vs_[depth] == u
    uint8_t pos[MAX_VCOUNT];
    for (uint8_t d = 0; d < query_.vcount_; d++)
        pos[indexing_orders_.vs_[d]] = d;

    uint8_t fn_count = 0;
    for (uint8_t d = 0; d < query_.vcount_; d++) {
        ext.fni_offs_[d] = fn_count;
        uint8_t u = indexing_orders_.vs_[d];
        for (uint8_t uu = 0; uu < query_.vcount_; uu++) {
            if (uu == u) continue;
            uint8_t eidx = query_.eidx_[u * query_.vcount_ + uu];
            if (eidx != UINT8_MAX && pos[uu] > d && fn_count < MAX_ECOUNT) {
                ext.fni_[fn_count] = pos[uu];
                ext.fni_eidx_[fn_count] = eidx;
                fn_count++;
            }
        }
    }
    ext.fni_offs_[query_.vcount_] = fn_count;
}

void Plan::GenerateRebuildFlagsWithOrder(
    uint8_t update_u0, uint8_t update_u1,
    std::initializer_list<uint8_t>&& starting_vertices,
    std::bitset<MAX_VCOUNT * MAX_VCOUNT>& rebuild_flags
) {
    rebuild_flags[update_u0 * query_.vcount_ + update_u1] = true;
    rebuild_flags[update_u1 * query_.vcount_ + update_u0] = true;

    std::vector<uint8_t> path(query_.vcount_, UINT8_MAX);
    for (const auto& starting_vertex: starting_vertices)
    {
        std::bitset<MAX_VCOUNT> visited;
        visited[starting_vertex] = true;
        path[0] = starting_vertex;
        std::vector<uint8_t> candidate_index(query_.vcount_, 0u);
        auto depth = 1u;

        while (true)
        {
            while (candidate_index[depth] < query_.nbrs_[path[depth - 1]].size())
            {
                const auto& next_v = query_.nbrs_[path[depth - 1]][candidate_index[depth]].first;
                if (visited[next_v])
                {
                    candidate_index[depth]++;
                    continue;
                }
                if (next_v == update_u0 || next_v == update_u1)
                {
                    // a path is found, set the rebuild flag
                    path[depth] = next_v;
                    for (auto i = 0u; i < depth; i++)
                    {
                        rebuild_flags[path[i] * query_.vcount_ + path[i + 1]] = true;
                        rebuild_flags[path[i + 1] * query_.vcount_ + path[i]] = true;
                    }
                    path[depth] = UINT8_MAX;
                    candidate_index[depth]++;
                    continue;
                }

                visited[next_v] = true;
                path[depth] = next_v;
                depth++;
            }
            if (candidate_index[depth] >= query_.nbrs_[path[depth - 1]].size())
            {
                candidate_index[depth] = 0u;
                depth--;
                if (depth == 0u) break;
                visited[path[depth]] = false;
                candidate_index[depth] ++;
            }
        }
    }
}

void Plan::GenerateRebuildVFlagsWithOrder(
    uint8_t update_u0, uint8_t update_u1,
    uint8_t index
) {
    // iterate over vertices on the indexing order
    for (auto i = 0u; i < query_.vcount_; i++)
    {
        const auto& u = indexing_orders_[index].vs_[i];
        if (i == 0u)
        {
            rebuild_v_flags_[index][u] = true;
            continue;
        }
        uint8_t sum = std::transform_reduce(
            &indexing_orders_[index].bni_[indexing_orders_[index].bni_offs_[i]],
            &indexing_orders_[index].bni_[indexing_orders_[index].bni_offs_[i + 1]],
            0,
            [](const auto& v1, const auto& v2){return v1 + v2;},
            [this, u, index](const auto& bni){
                return this->rebuild_flags_[index][this->indexing_orders_[index].vs_[bni] * this->query_.vcount_ + u] ? 1u : 0u;
            }
        );
        if (sum == 0u)
        {
            rebuild_v_flags_[index][u] = false;
        }
        else if (sum == indexing_orders_[index].bni_offs_[i + 1] - indexing_orders_[index].bni_offs_[i])
        {
            rebuild_v_flags_[index][u] = true;
        }
        else
        {
            std::cout << "Error in order generation!\n";
            exit(-1);
        }
    }
}

void Plan::GenerateCartesianProductInfo(
    std::vector<uint8_t>& cur_order,
    uint8_t index
) {
    CartesianProductType type;
    uint8_t vertex = cur_order[query_.vcount_ - 1];
    if (query_.nbrs_[vertex].size() == 1)
    {
        cartesian_product_info_[index][query_.vcount_ - 1] = CartesianProductType::TreeSingle;
        type = CartesianProductType::TreeCartesianProduct;
    }
    else
    {
        cartesian_product_info_[index][query_.vcount_ - 1] = CartesianProductType::NonTreeSingle;
        type = CartesianProductType::NonTreeCartesianProduct;
    }
    for (auto i = 1u; i < query_.vcount_ - 1; i++)
    {
        if (type == CartesianProductType::None)
        {
            cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::None;
        }
        else
        {
            vertex = cur_order[query_.vcount_ - 1 - i];
            bool valid = true;
            // check if there exist a forward neighbor
            for (auto j = 0u; j < i; j++)
            {
                uint8_t v_other = cur_order[query_.vcount_ - 1 - j];
                auto it = std::lower_bound(query_.nbrs_[vertex].begin(), query_.nbrs_[vertex].end(), make_pair<uint32_t, uint32_t>(v_other, 0));
                if (it != query_.nbrs_[vertex].end() && it->first == v_other)
                {
                    valid = false;
                    break;
                }
            }
            if (valid)
            {
                if (query_.nbrs_[vertex].size() == 1 && type == CartesianProductType::TreeCartesianProduct)
                {
                    cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::TreeCartesianProduct;
                }
                else
                {
                    cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::NonTreeCartesianProduct;
                    type = CartesianProductType::NonTreeCartesianProduct;
                }
            }
            else
            {
                cartesian_product_info_[index][query_.vcount_ - 1 - i] = CartesianProductType::None;
                type = CartesianProductType::None;
            }
        }
    }
}