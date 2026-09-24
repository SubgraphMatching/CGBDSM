
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <sys/stat.h> /* For stat() */
#include <tuple>
#include <unordered_map>

#include "utils/config.h"
#include "utils/constants.h"
#include "graph/graph.h"


QueryGraph::QueryGraph()
: vcount_(0u)
, ecount_(0u)
, vlabels_()
, nbrs_()
, qv_offs_(MAX_VCOUNT + 1)
, qv_nbrs_(MAX_ECOUNT * 2)
, NLF_(MAX_ECOUNT * 2, 0u)
, first_NL_(MAX_ECOUNT * 2, 0u)
, last_NL_(MAX_ECOUNT * 2, 0u)

, qe_list_(MAX_ECOUNT)
, qe_labels_(MAX_ECOUNT)
, qe_reversed_labels_(MAX_ECOUNT)

, eidx_(MAX_VCOUNT * MAX_VCOUNT, UINT8_MAX)
, qe_eidx_(MAX_ECOUNT)
{}

DataGraph::DataGraph()
: vcount_(0u)
, vlabels_()
, initial_edges_()
, updated_edges_()
{}


CPUGraphLoader::CPUGraphLoader(const std::string& query_path, QueryGraph& query_graph)
: query_(query_graph)
, vlabel_count_(0u)
, elabel_count_(0u)
, vlabel_map_()
, elabel_map_()
{
    if (!FileExist(query_path.c_str()))
    {
        std::cout << "Failed to open: " << query_path << std::endl;
        exit(-1);
    }

    // create the query graph
    std::ifstream ifs(query_path);

    char type;
    while (ifs >> type)
    {
        if (type == 't')
        {
            char temp1;
            uint temp2;
            ifs >> temp1 >> temp2;
        }
        else if (type == 'v')
        {
            uint vertex_id, label;
            ifs >> vertex_id >> label;

            if (vlabel_map_.find(label) == vlabel_map_.end())
            {
                vlabel_map_[label] = vlabel_map_.size();
            }
            label = vlabel_map_.at(label);

            if (vertex_id >= query_.vlabels_.size())
            {
                query_.vlabels_.resize(vertex_id + 1, NOT_EXIST);
                query_.vlabels_[vertex_id] = label;
                query_.nbrs_.resize(vertex_id + 1);
            }
            else if (query_.vlabels_[vertex_id] == NOT_EXIST)
            {
                query_.vlabels_[vertex_id] = label;
            }

            query_.vcount_ += 1;
        }
        else
        {
            uint from_id, to_id, label;
            ifs >> from_id >> to_id >> label;

            if (elabel_map_.find(label) == elabel_map_.end())
            {
                elabel_map_[label] = elabel_map_.size();
            }
            label = elabel_map_.at(label);

            // In the query graph, an adjacency array is sorted
            std::pair insert_pair(to_id, label);
            auto lower = std::lower_bound(query_.nbrs_[from_id].begin(), query_.nbrs_[from_id].end(), insert_pair);
            query_.nbrs_[from_id].insert(lower, insert_pair);

            insert_pair = std::make_pair(from_id, label);
            lower = std::lower_bound(query_.nbrs_[to_id].begin(), query_.nbrs_[to_id].end(), insert_pair);
            query_.nbrs_[to_id].insert(lower, insert_pair);

            query_.ecount_ += 1;
        }
    }
    ifs.close();
    if (query_.vcount_ < 2)
    {
        std::cout << "the number of query verticex should not be less than 2!" << std::endl;
        exit(-1);
    }
}

void CPUGraphLoader::SetQueryMeta()
{
    // set other meta data of the class
    QV_COUNT = query_.vcount_;
    QE_COUNT = query_.ecount_;
    if (QV_COUNT > MAX_VCOUNT || QE_COUNT > MAX_ECOUNT)
    {
        std::cout << "The query graph should have at most " << MAX_VCOUNT
        << " vertices and " << MAX_ECOUNT << " edges.\n";
        exit(-1);
    }
    vlabel_count_ = vlabel_map_.size();
    elabel_count_ = elabel_map_.size();

    uint32_t edge_pos = 0u;
    for (auto u = 0u; u < query_.vcount_; u++)
    {
        query_.qv_offs_[u] = edge_pos;
        for (const auto& [uu, _]: query_.nbrs_[u])
        {
            query_.eidx_[u * query_.vcount_ + uu] = edge_pos;
            query_.qv_nbrs_[edge_pos] = uu;
            edge_pos++;
        }
    }
    query_.qv_offs_[query_.vcount_] = edge_pos;

    for (auto u = 0u; u < QV_COUNT; u++)
    {
        std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t, uint32_t>> distinct_NLs;
        for (const auto& [uu, label]: query_.nbrs_[u])
        {
            if (distinct_NLs.find(label * QV_COUNT + query_.vlabels_[uu]) == distinct_NLs.end())
            {
                distinct_NLs[label * QV_COUNT + query_.vlabels_[uu]] = {query_.eidx_[u * QV_COUNT + uu], query_.eidx_[u * QV_COUNT + uu], 1};
            }
            else
            {
                std::get<1>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu])) = query_.eidx_[u * QV_COUNT + uu];
                std::get<2>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu])) += 1;
            }
        }
        for (const auto& [uu, label]: query_.nbrs_[u])
        {
            if (std::get<2>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu])) > 2)
            {
                query_.first_NL_[query_.eidx_[u * QV_COUNT + uu]] = std::get<0>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu]));
                query_.last_NL_[query_.eidx_[u * QV_COUNT + uu]] = std::get<1>(distinct_NLs.at(label * QV_COUNT + query_.vlabels_[uu]));
            }
            else
            {
                query_.first_NL_[query_.eidx_[u * QV_COUNT + uu]] = query_.eidx_[u * QV_COUNT + uu];
                query_.last_NL_[query_.eidx_[u * QV_COUNT + uu]] = query_.eidx_[u * QV_COUNT + uu];
            }
        }
        for (const auto& [NL, t]: distinct_NLs)
        {
            query_.NLF_[std::get<0>(t)] = std::get<2>(t);
        }
    }

    edge_pos = 0u;
    for (auto u = 0u; u < query_.vcount_; u++)
    {
        for (const auto& [uu, label]: query_.nbrs_[u])
        {
            if (u > uu) continue;

            query_.qe_list_[edge_pos] = {u, uu};
            query_.qe_labels_[edge_pos] = {query_.vlabels_[u], label, query_.vlabels_[uu]};
            query_.qe_reversed_labels_[edge_pos] = {query_.vlabels_[uu], label, query_.vlabels_[u]};
            query_.qe_eidx_[edge_pos] = {
                query_.eidx_[u * query_.vcount_ + uu], query_.eidx_[uu * query_.vcount_ + u]
            };
            edge_pos++;
        }
    }

    for (auto u = 0u; u < QV_COUNT; u++)
    {
        for (const auto& [uu, _]: query_.nbrs_[u])
        {
            std::cout << "[" << u << "," << uu << "]: " << static_cast<uint32_t>(query_.eidx_[u * QV_COUNT + uu]) << ' ';
        }
        std::cout << '\n';
    }
}

void CPUGraphLoader::LoadInitial(const std::string& data_path, DataGraph& data_graph)
{
    if (!FileExist(data_path.c_str()))
    {
        std::cout << "Failed to open: " << data_path << std::endl;
        exit(-1);
    }

    data_graph.initial_edges_.resize(QE_COUNT * 2);
    // create the data graph
    std::ifstream ifs(data_path);
    char type;
    while (ifs >> type)
    {
        if (type == 't')
        {
            char temp1;
            uint temp2;
            ifs >> temp1 >> temp2;
        }
        else if (type == 'v')
        {
            uint vertex_id, label;
            ifs >> vertex_id >> label;
            data_graph.vlabels_.resize(vertex_id + 1, NOT_EXIST);
            if (vlabel_map_.find(label) == vlabel_map_.end())
            {
                continue;
            }
            label = vlabel_map_[label];

            if (vertex_id >= data_graph.vlabels_.size())
            {
                data_graph.vlabels_[vertex_id] = label;
            }
            else if (data_graph.vlabels_[vertex_id] == NOT_EXIST)
            {
                data_graph.vlabels_[vertex_id] = label;
            }
        }
        else
        {
            uint from_id, to_id, label;
            ifs >> from_id >> to_id >> label;
            if (data_graph.vlabels_[from_id] == NOT_EXIST)
            {
                continue;
            }
            if (data_graph.vlabels_[to_id] == NOT_EXIST)
            {
                continue;
            }
            if (elabel_map_.find(label) == elabel_map_.end())
            {
                continue;
            }
            label = elabel_map_[label];

            std::tuple elabals = {data_graph.vlabels_[from_id], label, data_graph.vlabels_[to_id]};
            for (auto i = 0u; i < QE_COUNT; i++)
            {
                if (query_.qe_labels_[i] == elabals)
                {
                    // (from_id, to_id) matches qe_list_[i]
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].first.push_back(from_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].second.push_back(to_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].first.push_back(to_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].second.push_back(from_id);
                }
                if (query_.qe_reversed_labels_[i] == elabals)
                {
                    // (to_id, from_id) matches qe_list_[i]
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].first.push_back(to_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].first].second.push_back(from_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].first.push_back(from_id);
                    data_graph.initial_edges_[query_.qe_eidx_[i].second].second.push_back(to_id);
                }
            }
        }
    }
    DV_COUNT = data_graph.vcount_ = data_graph.vlabels_.size();
    ifs.close();
}

void CPUGraphLoader::LoadUpdate(const std::string& update_path, DataGraph& data_graph, const uint32_t batch_size)
{
    if (!FileExist(update_path.c_str()))
    {
        std::cout << "Failed to open: " << update_path << std::endl;
        exit(-1);
    }

    uint32_t num_updated_edges = 0u;
    // create the update stream
    std::ifstream ifs(update_path);
    char type;
    while (ifs >> type)
    {
        if (type == 't')
        {
            char temp1;
            uint temp2;
            ifs >> temp1 >> temp2;
        }
        else if (type == 'v')
        {
            std::cout << "vertex update.\n";
            exit(-1);
        }
        else
        {
            uint from_id, to_id, label;
            ifs >> from_id >> to_id >> label;
            if (num_updated_edges % batch_size == 0)
            {
                data_graph.updated_edges_.emplace_back();
                data_graph.updated_edges_.back().resize(QE_COUNT * 2);
            }
            num_updated_edges++;
            if (data_graph.vlabels_[from_id] == NOT_EXIST)
            {
                continue;
            }
            if (data_graph.vlabels_[to_id] == NOT_EXIST)
            {
                continue;
            }
            if (elabel_map_.find(label) == elabel_map_.end())
            {
                continue;
            }
            label = elabel_map_[label];

            std::tuple elabals = {data_graph.vlabels_[from_id], label, data_graph.vlabels_[to_id]};
            for (auto i = 0u; i < QE_COUNT; i++)
            {
                if (query_.qe_labels_[i] == elabals)
                {
                    // (from_id, to_id) matches qe_list_[i]
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].first.push_back(from_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].second.push_back(to_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].first.push_back(to_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].second.push_back(from_id);
                }
                if (query_.qe_reversed_labels_[i] == elabals)
                {
                    // (to_id, from_id) matches qe_list_[i]
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].first.push_back(to_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].first].second.push_back(from_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].first.push_back(from_id);
                    data_graph.updated_edges_.back()[query_.qe_eidx_[i].second].second.push_back(to_id);
                }
            }
        }
    }
    ifs.close();
}

size_t CPUGraphLoader::FileExist(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}
