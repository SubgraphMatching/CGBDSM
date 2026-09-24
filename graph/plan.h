#ifndef GRAPH_PLAN_H
#define GRAPH_PLAN_H

#include <array>
#include <bitset>
#include <cstdint>
#include <vector>
#include "utils/config.h"
#include "utils/nucleus/nd_interface.h"
#include "graph/graph.h"

class OrderPerEdge
{
public:
    uint8_t vs_[MAX_VCOUNT];
    uint8_t bni_offs_[MAX_VCOUNT + 1];
    uint8_t bni_[MAX_ECOUNT]; // sorted by the reversed order of index
};

constexpr uint8_t MAX_TRI_PAIRS = 64;

struct BNPairTriInfo {
    uint8_t bn_i;     // offset into bni_ for first BN of the pair
    uint8_t bn_j;     // offset into bni_ for second BN of the pair
    uint8_t eidx_ij;  // directed edge index u_i -> u_j
};

struct IndexingOrderExt {
    // Triangle pair metadata
    uint8_t tri_offs_[MAX_VCOUNT + 1];
    BNPairTriInfo tri_pairs_[MAX_TRI_PAIRS];
    // Forward neighbor metadata
    uint8_t fni_offs_[MAX_VCOUNT + 1];
    uint8_t fni_[MAX_ECOUNT];
    uint8_t fni_eidx_[MAX_ECOUNT];
};

class Plan
{
private:
public:
    const QueryGraph& query_;
    enum CartesianProductType : uint8_t {
        None = 0,
        TreeSingle,
        NonTreeSingle,
        TreeCartesianProduct,
        NonTreeCartesianProduct
    };
    std::bitset<MAX_VCOUNT * MAX_VCOUNT> rebuild_flags_[MAX_ECOUNT];
    std::bitset<MAX_VCOUNT> rebuild_v_flags_[MAX_ECOUNT];
    std::array<CartesianProductType, MAX_VCOUNT> cartesian_product_info_[MAX_ECOUNT];
    OrderPerEdge indexing_orders_[MAX_ECOUNT];
    OrderPerEdge orders_[MAX_ECOUNT];
    IndexingOrderExt indexing_ext_[MAX_ECOUNT];

#ifdef USE_MERGED_MATCHING
    OrderPerEdge global_order_;
    std::array<CartesianProductType, MAX_VCOUNT> global_cartesian_product_info_;
#endif

    Plan(const QueryGraph& query_graph);

    void GenerateIndexingOrders_v2();
    // generate orders with the RI's ordering method
    void GenerateMatchingOrders_v2(uint32_t *cardinalities, float *avg_degrees);
#ifdef USE_MERGED_MATCHING
    void GenerateGlobalMatchingOrder(uint32_t *cardinalities, float *avg_degrees);
    void GenerateGlobalMatchingOrderGSI(uint32_t *cardinalities, float *avg_degrees);
#endif

    void PrintOrders();
private:
    void GroupDenseVertices(
        std::vector<nd_tree_node>& k34_tree,
        std::vector<uint8_t>& vrole
    );
    void AddVertices(
        std::vector<uint8_t>& order,
        std::bitset<MAX_VCOUNT>& visited,
        const std::vector<uint8_t>& vrole,
        const uint8_t group_id
    );
    void AddVertices(
        std::vector<uint8_t>& order,
        std::bitset<MAX_VCOUNT>& visited,
        const std::vector<uint8_t>& vrole
    );
    void BuildIndexingOrder(
        const uint8_t u0, const uint8_t u1,
        OrderPerEdge& indexing_orders_,
        IndexingOrderExt& ext
    );
    void GenerateRebuildFlagsWithOrder(
        uint8_t update_u0, uint8_t update_u1,
        std::initializer_list<uint8_t>&& starting_vertices,
        std::bitset<MAX_VCOUNT * MAX_VCOUNT>& rebuild_flags
    );
    void GenerateRebuildVFlagsWithOrder(
        uint8_t update_u0, uint8_t update_u1,
        uint8_t index
    );
    void GenerateCartesianProductInfo(
        std::vector<uint8_t>& cur_order,
        uint8_t index
    );
#ifdef USE_MERGED_MATCHING
    void GenerateCartesianProductInfoGlobal(
        std::vector<uint8_t>& cur_order,
        std::array<CartesianProductType, MAX_VCOUNT>& cp_info
    );
    std::pair<uint8_t, uint8_t> SelectStartingEdge(uint32_t *cardinalities, float *avg_degrees);
#endif
};

#endif
