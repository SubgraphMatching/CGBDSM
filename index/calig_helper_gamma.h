#ifndef MATCHING_CALIG_HELPER_GAMMA
#define MATCHING_CALIG_HELPER_GAMMA

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
#ifdef ENABLE_CPU_DFS
#include "index/cpu_index_mirror.h"
#endif

class CaLiGHelperGamma {
    CaLiG *calig;
    RelationsGPU* gpu_index;
    RelationsGPU* gpu_update_index;
    RelationsGPU* gpu_local_update_index;
    uint32_t num_edges = 0;

    MemPool<uint32_t>* nbr_mem_pool_;  // borrowed from GPUGraphLoader
#ifdef ENABLE_CPU_DFS
    CPUIndexMirror* cpu_mirror_ = nullptr;  // owned; host mirror of the GPU index
#endif

    ska::flat_hash_map<uint32_t, uint32_t> global_vid2newidx;
    std::vector<uint32_t> dense_vid_map_;  // direct array: old_vid -> new_idx (UINT32_MAX if unmapped)
    std::vector<std::pair<uint32_t, uint32_t>> valid_vid_pairs_;

    Plan *plan;

    // Per-thread temporary edge buffers (64-bit encoded, restored from calig_helper.cu)
    std::vector<thrust::host_vector<uint64_t>> tmp_el;

    // For initial build only (ConvertGlobalIndex Phase A)
    thrust::device_vector<uint64_t> el_dev;
    thrust::device_vector<uint32_t> el_32_dev;

    // For TransferUpdateIndex (flat CSR rebuild each batch)
    thrust::device_vector<uint64_t> el_update_dev;
    thrust::device_vector<uint32_t> el_update_32_dev;

    // For incremental merge (reused across batches)
    thrust::device_vector<uint64_t> tmp_sort_dev_;       // sorted edge staging
    thrust::device_vector<uint32_t> tmp_merge_32_;       // extracted v values for merge

    // Reusable host buffer for UpdateGlobalIndex (avoids per-call allocation)
    thrust::host_vector<uint64_t> update_host_enc_;

    // Per-batch staging buffers (64-bit encoded, restored from calig_helper.cu)
    std::vector<std::vector<uint64_t>> el_host_stage_;
    std::vector<std::vector<uint64_t>> del_host_stage_;
    std::vector<std::vector<uint64_t>> el_update_host_stage_;

    // Per-direction update pairs with compressed IDs (for UpdateGlobalIndex)
    std::vector<std::vector<std::vector<std::pair<uint32_t, uint32_t>>>> G_UPDATE_FLAT_;

    std::vector<bool> vertex_mapping_changed_;
    std::vector<uint32_t> num_valid_vertices_buf_;

    // GPU capabilities for sizes_/nbrs_/capability_ pointer arrays
    std::vector<size_t> gpu_capabilities_;

    // Per-batch edge emptiness flags (set by PrepareUpdateIndex, read by Stage 3)
    std::vector<std::array<bool, MAX_ECOUNT>> edge_ok_per_batch_;

    // CPU-only vertex mapping update (uses LI snapshot)
    void UpdateVertexMappingCPU(uint32_t batch);
    // GPU-side allocation from updated vertex mapping
    void AllocateFromMappingGPU(uint32_t batch);

    void buildGlobalVertexMapping();
    uint32_t getNewVertexId(uint32_t old_vid);

    // Core method: sort 64-bit edges → generateSizePtr flat CSR → mergeCSROntoGraph
    void mergeSortedEdgesToIndex(
        thrust::device_vector<uint64_t>& sorted_edges,
        bool is_insertion,
        uint32_t batch);

public:
    CaLiGHelperGamma(CaLiG *calig, RelationsGPU* gpu_index, RelationsGPU* gpu_update_index,
                     RelationsGPU* gpu_local_update_index, Plan *plan,
                     uint32_t num_batches, MemPool<uint32_t>* nbr_mem_pool);
    ~CaLiGHelperGamma();
    void ConvertGlobalIndex(uint32_t *cardinalities, float *degrees);

    void UpdateGlobalIndex(uint32_t batch, uint8_t cur_i);
    void BatchUpdateGlobalIndex(uint32_t batch);  // merge ALL query edge updates in one pass

    // CPU-parallel phase: scan CaLiG data -> host vectors (no GPU access)
    void PrepareUpdateAll(uint32_t batch);
    void PrepareUpdateIndex(uint32_t batch);

    // GPU-transfer phase: host vector -> GPU
    void TransferUpdateAll(uint32_t batch);
    bool TransferUpdateIndex(uint32_t batch);
    void PrintUpdateIndexSizes(uint32_t batch);

    uint32_t getNumValidVertices(uint32_t batch) const { return num_valid_vertices_buf_[batch]; }
    const bool* GetEdgeOk(uint32_t batch) const { return edge_ok_per_batch_[batch].data(); }
#ifdef ENABLE_CPU_DFS
    // Host-side vector mirror of the GPU index (NUMA-2 bound in Phase B),
    // kept in sync via sorted merge from the same host staging edge-lists.
    CPUIndexMirror* GetCPUMirror() { return cpu_mirror_; }
    // Accumulated mirror-op timings (ms) for profiling.
    double cpu_mirror_build_ms_ = 0.0;  // ConvertGlobalIndex init
    double cpu_mirror_ins_ms_ = 0.0;     // TransferUpdateAll insert
    double cpu_mirror_del_ms_ = 0.0;     // TransferUpdateAll delete
    double cpu_mirror_rebuild_ms_ = 0.0; // TransferUpdateIndex rebuild update_
    double cpu_mirror_batch_ms_ = 0.0;   // BatchUpdateGlobalIndex merge
    double GetCPUMirrorBuildMs() const { return cpu_mirror_build_ms_; }
    double GetCPUMirrorInsMs() const { return cpu_mirror_ins_ms_; }
    double GetCPUMirrorDelMs() const { return cpu_mirror_del_ms_; }
    double GetCPUMirrorRebuildMs() const { return cpu_mirror_rebuild_ms_; }
    double GetCPUMirrorBatchMs() const { return cpu_mirror_batch_ms_; }
#endif

    // Map CaLiG's per-(ei, update-edge) edge_conflict into per-(ei, compressed data vertex)
    // for upload to GPU g_d_cf. A vertex is 0 (cannot-skip) on ei iff it is an endpoint of an
    // update edge that conflicts on ei; default 1 (conflict-free). Size = QE_COUNT * compressed_DV.
    // std::vector<uint8_t> BuildCompressedConflictFree2D();
};


#endif // MATCHING_CALIG_HELPER_GAMMA
