#ifndef GRAPH_GRAPH_GPU
#define GRAPH_GRAPH_GPU

#include <cstdint>
#include <vector>

#include "utils/config.h"
#include "utils/types.h"
#include "utils/mem_pool.h"

#include "graph/graph.h"
#include "graph/graph_gpu.h"
#include "graph/plan.h"
#ifdef ENABLE_CPU_DFS
#include <tbb/task_arena.h>
class CPUIndexMirror;  // host-side mirror of the index (defined in index/cpu_index_mirror.h)
#endif

class ValidBits
{
public:
    uint32_t *bits_[MAX_VCOUNT] = {};
};

extern __constant__ OrderPerEdge C_INDEXING_ORDERS[MAX_ECOUNT];
extern __constant__ OrderPerEdge C_ORDERS[MAX_ECOUNT];
extern __constant__ IndexingOrderExt C_INDEXING_ORDERS_EXT[MAX_ECOUNT];
extern __constant__ ValidBits C_VALID_BITS;
extern __constant__ uint8_t C_DIR_TO_EDGE[MAX_ECOUNT * 2];
extern __constant__ uint8_t C_REBUILD_V_FLAGS[MAX_ECOUNT * MAX_VCOUNT];
extern __constant__ float C_AVG_DEGREES[MAX_ECOUNT * 2];

#ifdef USE_MERGED_MATCHING
extern __constant__ OrderPerEdge C_GLOBAL_ORDER;
extern __constant__ uint8_t C_GLOBAL_CP_INFO[MAX_VCOUNT];
extern __constant__ uint8_t C_EDGE_ENDPOINTS[MAX_ECOUNT][2];
extern __constant__ uint8_t C_QV_REPR_EDGE[MAX_VCOUNT];
extern __constant__ uint8_t C_QV_REPR_EP[MAX_VCOUNT];
#endif

class RelationsGPU
{
public:
    uint32_t **nbrs_[MAX_ECOUNT * 2];
    uint32_t *capability_[MAX_ECOUNT * 2];
    uint32_t *sizes_[MAX_ECOUNT * 2];
public:
    RelationsGPU();
};

class CandidatesGPU
{
public:
    uint32_t *candidate_bits_[MAX_VCOUNT];
public:
    CandidatesGPU();
};

class GPUGraphLoader
{
private:
    const QueryGraph& query_;
    const Plan& plan_;

    // For CUB
    void *d_temp_storage_;
    size_t temp_storage_bytes_;
    size_t temp_storage_capability_;

    // For indexing
    bool *cand_flag_;
    uint32_t cand_flag_capability_;
    uint32_t *d_new_cand_count_[2];
    Tries temp_tries_[2];
    TrieCapability temp_tries_capability_[2];
    uint32_t *helper_relation_[2];
    uint32_t helper_relation_capability_[2];

    uint32_t *local_nbr_[MAX_ECOUNT * 2];
    uint32_t local_nbr_capability_[MAX_ECOUNT * 2];

    uint32_t *cum_bn_;

    // For Valid Bits
    ValidBits valid_bits_;
    uint32_t valid_bits_capability_[MAX_VCOUNT] = {};

    // For Valid Bits (per-edge, merged) — 3D: [edge_e][match_ei][endpoint]
    uint32_t* flat_edge_bits_ = nullptr;
    uint32_t flat_edge_bits_size_ = 0;
    uint32_t** d_edge_vb_ptrs_ = nullptr;

    // For Support Masks (per-edge, merged) — 3D: [edge_e][query_v][data_v] smask_t
    // Replaces 1-bit bitmap with multi-bit support mask per vertex for false-positive reduction
    smask_t* flat_support_masks_ = nullptr;
    uint32_t flat_support_masks_size_ = 0;  // size in smask_t units
    smask_t** d_edge_sm_ptrs_ = nullptr;

    // For Support Masks Temp (per-edge, merged) — push-all temp buffer
    // Same layout as flat_support_masks_: [edge_e][query_v][data_v]
    smask_t* flat_support_masks_tmp_ = nullptr;
    uint32_t flat_support_masks_tmp_size_ = 0;
    smask_t** d_edge_sm_tmp_ptrs_ = nullptr;

    // For Enumeration
    unsigned long long int res_;
    unsigned long long int res_size_;

    unsigned long long int new_res_;
    unsigned long long int *new_res_size_;
    unsigned long long int h_new_res_size_;
    unsigned long long int h_max_new_res_size_;

    uint8_t cur_depth_;
    uint8_t new_depth_;
#ifdef ENABLE_CPU_DFS
    bool last_step5_cartesian_ = false;  // set in MatchingBitAll Step 5
    // ---- Phase B: CPU/GPU final-DFS split ----
    CPUIndexMirror* cpu_mirror_ = nullptr;       // not owned (owned by CaLiGHelperGamma)
    tbb::task_arena* cpu_dfs_arena_ = nullptr;   // NUMA-2 bound; owned
    float cpu_dfs_ratio_ = 0.3f;                 // EMA: CPU's share of the Step-5 frontier
    double cpu_dfs_alpha_ = 0.3;                 // EMA smoothing factor
    uint32_t* h_frontier_buf_ = nullptr;         // reusable host frontier buffer
    size_t h_frontier_buf_cap_ = 0;
    cudaStream_t cpu_dfs_stream_ = nullptr;      // side stream: overlaps frontier D2H with the GPU launch
    cudaEvent_t cpu_dfs_ev0_ = nullptr, cpu_dfs_ev1_ = nullptr;
    cudaStream_t cpu_mask_stream_ = nullptr;  // D2H of the support mask (decoupled from the frontier D2H stream)
    unsigned long long cpu_dfs_min_rows_ = 8192; // below this, keep Step 5 fully on GPU
#endif

    // For memory allocation
    MemPool<uint32_t> nbr_mem_pool_;
    CyclicQueue<uint32_t> res_queue_;
    unsigned long *res_size_cartesian_product_;
    unsigned long *max_res_size_cartesian_product_;
    RelationsGPU local_index_base_;

#ifdef USE_MERGED_MATCHING
    // For merged matching
    RelationsGPU* d_all_local_ = nullptr;  // device array [QE_COUNT]
    bool* d_edge_ok_ = nullptr;
    // uint8_t* d_cf_ = nullptr;     // device [compressed DV_COUNT], per-data-vertex conflict-free
    // size_t d_cf_cap_ = 0;
    // bool d_cf_sym_set_ = false;
#endif

#ifdef USE_GLOBAL_RQ
    uint32_t* d_global_rq_ = nullptr;
    unsigned long long global_rq_capability_ = 0;
#endif
public:
GPUGraphLoader(
        const CPUGraphLoader& cpu_loader, 
        const QueryGraph& query,
        const Plan& plan
    );
    ~GPUGraphLoader();
    void LoadQuery();
    void LoadPlan();
    void PrintGammaMetrics(const RelationsGPU& index_gpu);
    void BuildTries(const EdgeBatch& edge_lists, Tries tries[], const uint8_t i);
    void DeallocTries(Tries tries[]);
    void AllocRelations(
        const DataGraph& data_graph, const Tries tries[],
        RelationsGPU& data_graph_gpu, RelationsGPU& index_gpu, 
        CandidatesGPU& candidates_gpu
    );
    void AllocOnline();
    void AllocRelations(uint32_t DV_COUNT_);
    void ReAllocValidBits();
    bool BuildLocalIndex(const RelationsGPU& index_gpu, RelationsGPU& local_index, const RelationsGPU& update_index, const uint8_t cur_i, const float *avg_degrees);
    bool BuildLocalIndexBit(const RelationsGPU& index_gpu, RelationsGPU& local_index, const RelationsGPU& update_index, const uint8_t cur_i, const float *avg_degrees);
    void BuildLocalIndexBitAll(const RelationsGPU& index_gpu, const RelationsGPU& update_index, const float *avg_degrees);
    void SetupLocalIndex(const RelationsGPU& index_gpu, RelationsGPU& local_index, const RelationsGPU& update_index, const uint8_t cur_i);
    void SetConstantValidBits(uint8_t edge_idx);
    void Matching(RelationsGPU& local_index, const uint8_t i, unsigned long long int& num_matches);
    void MatchingBit(RelationsGPU& local_index, const uint8_t i, unsigned long long int& num_matches);
#ifdef USE_MERGED_MATCHING
    void SetupLocalIndexAll(const RelationsGPU& index_gpu, const RelationsGPU& update_index);
    void MatchingBitAll(const RelationsGPU& index_gpu, const RelationsGPU& update_index, const bool* edge_ok, unsigned long long int& num_matches);
//     // Upload per-data-vertex (compressed-ID) conflict-free flags; size = compressed DV_COUNT.
//     void UploadConflictFree(const uint8_t* cf, uint32_t size);
// #if RELAX_DEBUG
//     // D2H-read and print the relax "skip-but-binary-would-fail" counter (cumulative).
//     void RelaxDbgPrintSkipFail();
// #endif
#endif

    MemPool<uint32_t>& getNbrMemPool() { return nbr_mem_pool_; }
#ifdef ENABLE_CPU_DFS
    // Phase A correctness cross-check: expose the final-DFS frontier so the host
    // can re-count on the CPU mirror and compare to the GPU's batch count.
    // res_/res_size_/cur_depth_ are left holding Step-5's input frontier after
    // MatchingBitAll returns (write_res=false does not overwrite it).
    unsigned long long cpuGetResOffset() const { return res_; }
    unsigned long long cpuGetResSize() const { return res_size_; }
    uint8_t cpuGetCurDepth() const { return cur_depth_; }
    const uint32_t* cpuGetResQueueArray() const { return res_queue_.array_; }
    unsigned long long cpuGetResQueueCapability() const { return res_queue_.capability_; }
    bool cpuLastStep5Cartesian() const { return last_step5_cartesian_; }
    // Phase B: wire up the host mirror + a NUMA-bound TBB arena for CPU DFS.
    // avoid_node: NUMA node used by Stage-1 (CPU DFS picks a different one).
    void InitCPUDfs(CPUIndexMirror* mirror, int numa_node, int avoid_node);
    // D2H flat_support_masks_ into the CPU mirror on the side stream (overlaps
    // the GPU BFS steps); used by the CPU DFS as a cheap candidate prune.
    void SyncCPUMirrorMasks();
    // Split Step 5's final DFS: CPU takes frontier [0,k), GPU takes [k,res_size_),
    // run concurrently, join, and update the EMA ratio. Returns true if handled.
    bool Step5CPUGPUSplit(const RelationsGPU& update_index, unsigned long long int& num_matches);
    float GetCPUDfsRatio() const { return cpu_dfs_ratio_; }
    tbb::task_arena* GetCPUDfsArena() { return cpu_dfs_arena_; }
#endif
};

#endif