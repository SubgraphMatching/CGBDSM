#ifndef GRAPH_GRAPH
#define GRAPH_GRAPH

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "utils/types.h"

class CPUGraphLoader;
class GPUGraphLoader;
class Plan;

class QueryGraph
{
private:
    std::vector<uint32_t> vlabels_;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> nbrs_;
    // CSR of the query graph
    std::vector<uint8_t> qv_offs_;
    std::vector<uint8_t> qv_nbrs_;
    // NLF array
    std::vector<uint8_t> NLF_;
    std::vector<uint8_t> first_NL_;
    std::vector<uint8_t> last_NL_;

    // list of the query edges and their labels (direction not considered)
    // used to enumerate query edges

    // find the relation index (direction considered) based on the two endpoints
    // find the relation index (direction considered) based on the edge index in qe_list_
public:
    std::vector<std::pair<uint32_t, uint32_t>> qe_list_;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> qe_labels_;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> qe_reversed_labels_;
    std::vector<std::pair<uint32_t, uint32_t>> qe_eidx_;
    uint32_t vcount_;
    uint32_t ecount_;
    std::vector<uint8_t> eidx_;
    QueryGraph();
    friend class CPUGraphLoader;
    friend class GPUGraphLoader;
    friend class Plan;
};

class DataGraph
{
private:
    uint32_t vcount_;
    std::vector<uint32_t> vlabels_;
public:
    EdgeBatch initial_edges_;
    std::vector<EdgeBatch> updated_edges_;

public:
    DataGraph();
    friend class CPUGraphLoader;
    friend class GPUGraphLoader;
};

class CPUGraphLoader
{
private:
    QueryGraph& query_;
    uint32_t vlabel_count_;
    uint32_t elabel_count_;
    std::unordered_map<uint32_t, uint32_t> vlabel_map_;
    std::unordered_map<uint32_t, uint32_t> elabel_map_;

public:
    CPUGraphLoader(const std::string& query_path, QueryGraph& query_graph);
    void SetQueryMeta();
    void LoadInitial(const std::string& data_path, DataGraph& data_graph);
    void LoadUpdate(const std::string& update_path, DataGraph& data_graph, const uint32_t batch_size);

private:
    size_t FileExist(const char *path);
    friend class GPUGraphLoader;
};


#endif
