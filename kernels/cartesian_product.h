
#include <cstdint>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/graph_gpu.h"

__global__ void GetNumTree(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t num_vs
);

__global__ void enumerateCartesianProductTree(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size,
    const uint8_t num_vs
);

__global__ void GetNumTreeBit(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t num_vs,
    smask_t** sm_ptrs,
    const uint8_t ei
);

// Support mask variants
__global__ void enumerateCartesianProductTreeBit(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size,
    const uint8_t num_vs,
    smask_t** sm_ptrs,
    const uint8_t ei
);

#ifdef USE_MERGED_MATCHING
__global__ void GetNumTreeAllBit(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU* d_all_local,
    const RelationsGPU update_index,
    smask_t** edge_sm_ptrs,
    const uint8_t num_vs
);

__global__ void enumerateCartesianProductAllBit(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU* d_all_local,
    const RelationsGPU update_index,
    smask_t** edge_sm_ptrs,
    unsigned long long *new_res_size,
    const uint8_t num_vs
);
#endif
