#ifndef KERNELS_ENUMERATION
#define KERNELS_ENUMERATION

#include <cstdint>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/graph_gpu.h"

__global__ void write_initial_partial_results(
    RelationsGPU index_gpu,
    const uint8_t idx,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_new_res_size
);

__global__ void write_initial_partial_results_bit(
    RelationsGPU index_gpu,
    const uint8_t idx,
    const uint8_t qv0,
    const uint8_t qv1,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size,
    smask_t** sm_ptrs,
    const uint8_t ei
);

__global__ void write_cpu_partial_results(
    const uint32_t* cpu_results,
    const uint32_t num_cpu_results,
    const uint32_t start_depth,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
);

__global__ void extendDFS(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const RelationsGPU index,
    const uint8_t oi
);

__global__ void extendBFSRegTwo(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth_arg,
    const bool write_res
);

__global__ void extendBFSShareAll(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth_arg,
    const bool write_res
);

__global__ void extendBFSDFSShareZero(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

__global__ void extendBFSDFSRegTwo(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

__global__ void extendBFSDFSShareTwo(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

__global__ void extendBFSDFSShareAll(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

#define DynMemSize(start_depth, end_depth)              \
    start_depth * NWARP_PER_BLOCK * sizeof(uint32_t) +  \
    (end_depth - start_depth) * NWARP_PER_BLOCK * (     \
        sizeof(uint32_t) * WARP_SIZE                    \
        + sizeof(uint8_t) * 3 + sizeof(uint32_t)        \
        + sizeof(bool) + sizeof(uint8_t)                \
    ) + NWARP_PER_BLOCK * WARP_SIZE + NWARP_PER_BLOCK

__global__ void extendBFSDFSDynamicShared(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res
);

// Support mask variant: same as extendBFSDFSRegTwo but with smask_t support mask + path mask filtering
__global__ void extendBFSDFSRegTwoBit(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const uint8_t end_depth,
    const bool write_res,
    smask_t** sm_ptrs,
    const uint8_t ei
);

#ifdef USE_MERGED_MATCHING
__global__ void writeInitialResultsAllBit(
    const RelationsGPU* d_all_local,
    const RelationsGPU d_update_index,
    smask_t** edge_sm_ptrs,
    const bool* edge_ok,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
);

__global__ void extendBFSAllBit(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU* d_all_local,
    const RelationsGPU update_index,
    smask_t** edge_sm_ptrs,
    uint8_t start_depth,
    uint8_t end_depth,
    bool write_res
);

#ifdef USE_GLOBAL_RQ
__global__ void extendBFSAllBitGlobal(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU* d_all_local,
    const RelationsGPU update_index,
    smask_t** edge_sm_ptrs,
    uint8_t start_depth,
    uint8_t end_depth,
    bool write_res,
    uint32_t* __restrict__ global_rq
);
#endif
#endif

#endif