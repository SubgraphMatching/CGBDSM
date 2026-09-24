#pragma once

#include <cstdint>
#include "gpma/gpma.cuh"

// Wraps a single Multi_GPMA with one block per query edge direction.
// Keys encode the edge direction in the high 5 bits of the upper 32 bits:
//   key = ((e << 27 | new_u) << 32) | new_v
// Block e stores all candidate edges for query edge direction e.
class CGCSM_GPMA {
public:
    void Init(uint32_t num_edges, uint32_t max_vertices);

    // Bulk-load all edges at once.
    // all_keys: device array of merged-encoded keys, sorted.
    // counts_per_edge: host array of edge counts per direction (num_edges entries).
    void BulkLoad(const uint64_t* d_all_keys, uint32_t total_count,
                  const uint32_t* counts_per_edge);

    // Batch insert and delete across all directions at once.
    // Keys use merged encoding: ((e << 27 | new_u) << 32) | new_v
    // block_delta: host array of num_edges entries, edge count change per direction (can be negative)
    void BatchUpdate(uint64_t* d_ins_keys, uint32_t ins_count,
                     uint64_t* d_del_keys, uint32_t del_count,
                     int32_t* block_delta = nullptr,
                     uint32_t new_max_vertices = 0);

    // Extract all valid edges for one gamma_eidx into a flat sorted array.
    // Output keys are in original encoding: (new_u << 32 | new_v).
    uint32_t ExtractEdges(uint32_t gamma_eidx, uint64_t* d_out_keys);

    void Destroy();

    uint32_t num_edges() const { return num_edges_; }
    uint32_t max_vertices() const { return max_vertices_; }

private:
    Multi_GPMA* gpma_ = nullptr;
    uint32_t num_edges_ = 0;    // QE_COUNT * 2
    uint32_t max_vertices_ = 0;

    void ensureUpdateBuffer(uint32_t needed);

    // Capacity trackers for ReAlloc (number of elements, not bytes)
    size_t cap_update_keys_ = 0;
    size_t cap_update_values_ = 0;
    size_t cap_update_nodes_ = 0;
    size_t cap_unique_update_nodes_ = 0;
    size_t cap_update_offset_ = 0;
    size_t cap_tmp_keys_ = 0;
    size_t cap_tmp_values_ = 0;
    size_t cap_tmp_label_ = 0;
    size_t cap_tmp_exscan_ = 0;
};
