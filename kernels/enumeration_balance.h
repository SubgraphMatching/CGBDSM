#ifndef KERNELS_ENUMERATION_BALANCE
#define KERNELS_ENUMERATION_BALANCE

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/graph_gpu.h"

#ifdef USE_MERGED_MATCHING

__global__ void extendBFSAllBitBalance(
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

#endif // USE_MERGED_MATCHING

#endif // KERNELS_ENUMERATION_BALANCE
