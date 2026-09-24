#ifndef MATCHING_CALIG_HELPER
#define MATCHING_CALIG_HELPER

#include <vector>
#include <array>
#include <iostream>
#include <cstdio>
#include <unordered_map>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include "index/calig.h"
#include "graph/graph_gpu.h"

class CaLiGHelper {
    CaLiG *calig;
    RelationsGPU* gpu_index;
    RelationsGPU* gpu_update_index;
    RelationsGPU* gpu_local_update_index;
    uint32_t num_edges = 0;

    ska::flat_hash_map<uint32_t, uint32_t> global_vid2newidx;
    std::vector<std::pair<uint32_t, uint32_t>> valid_vid_pairs_;

    std::vector<thrust::host_vector<uint64_t>> tmp_el;
    thrust::host_vector<uint64_t> el, el_update, el_updated;
    thrust::device_vector<uint64_t> el_dev, el_update_dev, el_updated_dev;
    thrust::device_vector<uint32_t> el_32_dev, el_update_32_dev, el_updated_32_dev;
    Plan *plan;

    // Per-batch staging buffers
    std::vector<std::vector<uint64_t>> el_host_stage_;
    std::vector<std::vector<uint64_t>> del_host_stage_;
    std::vector<std::vector<uint64_t>> el_update_host_stage_;
    std::vector<std::vector<std::vector<std::pair<uint32_t, uint32_t>>>> G_UPDATE_FLAT_;
    std::vector<bool> vertex_mapping_changed_;
    std::vector<uint32_t> num_valid_vertices_buf_;

    std::vector<size_t> gpu_capabilities_;

    // Per-batch edge emptiness flags (set by PrepareUpdateIndex, read by Stage 3)
    std::vector<std::array<bool, MAX_ECOUNT>> edge_ok_per_batch_;

    // CPU-only vertex mapping update (uses LI snapshot)
    void UpdateVertexMappingCPU(uint32_t batch);
    // GPU-side allocation from updated vertex mapping
    void AllocateFromMappingGPU(uint32_t batch);

    void buildGlobalVertexMapping();
    uint32_t getNewVertexId(uint32_t old_vid);

public:
    CaLiGHelper(CaLiG *calig, RelationsGPU* gpu_index, RelationsGPU* gpu_update_index, RelationsGPU* gpu_local_update_index, Plan *plan, uint32_t num_batches);
    ~CaLiGHelper();
    void ConvertGlobalIndex(uint32_t *cardinalities, float *degrees);

    void UpdateGlobalIndex(uint32_t batch, uint8_t cur_i);

    // CPU-parallel phase: scan CaLiG data -> host vectors (no GPU access)
    void PrepareUpdateAll(uint32_t batch);
    void PrepareUpdateIndex(uint32_t batch);

    // GPU-transfer phase: host vector -> GPU
    void TransferUpdateAll(uint32_t batch);
    bool TransferUpdateIndex(uint32_t batch);

    uint32_t getNumValidVertices(uint32_t batch) const { return num_valid_vertices_buf_[batch]; }
    const bool* GetEdgeOk(uint32_t batch) const { return edge_ok_per_batch_[batch].data(); }

    // std::vector<uint8_t> BuildCompressedConflictFree2D();
};


#endif // MATCHING_CALIG_HELPER
