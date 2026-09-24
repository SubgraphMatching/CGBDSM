#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/graph_gpu.h"

__global__ void GetNumTree(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t num_vs
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (auto i = tid; i < res_size; i += num_threads)
    {
        max_num_matches[i] = 1ul;
        for (uint8_t j = 0u; j < num_vs; j++)
        {
            const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - num_vs + j]];
            const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j]];
            const uint32_t& v = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - num_vs) + pre_qv_idx) % C_RES_QUEUE.capability_];

            max_num_matches[i] *= index.sizes_[pre_qe_idx][v];
        }
    }
}

__global__ void enumerateCartesianProductTree(
    const unsigned long long res,
    const unsigned long long res_size,
    const unsigned long *max_num_matches,
    const unsigned long *total_i,
    const RelationsGPU index,
    const uint8_t oi,
    unsigned long long *new_res_size,
    const uint8_t num_vs
) {
    __shared__ uint32_t visited[NWARP_PER_BLOCK][WARP_SIZE][MAX_VCOUNT - 1];
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    unsigned long long res_index;
    uint32_t map_v;
    unsigned long res_index_index;
    for (auto i = gwarp_id; i < DIV_CEIL(*total_i, WARP_SIZE); i += num_warps)
    {
        // binary search
        bool found = true;
        const unsigned long ii = i * WARP_SIZE + lane_id;
        if (ii >= *total_i) found = false;
        if (found)
        {
            res_index = lower_bound(max_num_matches + 1, res_size, ii + 1);

            res_index_index = ii - max_num_matches[res_index];
            for (uint8_t k = 0u; k < C_QV_COUNT - num_vs; k++)
            {
                visited[warp_id][lane_id][k] = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs) + k) % C_RES_QUEUE.capability_];
            }
        }
        __syncwarp();
        for (uint8_t j = 0u; j < num_vs; j++)
        {
            if (found)
            {
                const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - num_vs + j]];
                const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j]];
                const uint32_t& v = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs) + pre_qv_idx) % C_RES_QUEUE.capability_];

                map_v = index.nbrs_[pre_qe_idx][v][res_index_index % index.sizes_[pre_qe_idx][v]];
                res_index_index /= index.sizes_[pre_qe_idx][v];

                for (uint8_t k = 0u; k < C_QV_COUNT - num_vs + j; k++)
                {
                    if (map_v == visited[warp_id][lane_id][k])
                    {
                        found = false;
                        break;
                    }
                }
                if (!found) break;
                if (j != num_vs - 1) visited[warp_id][lane_id][C_QV_COUNT - num_vs + j] = map_v;
            }
        }
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0)
        {
            atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}

__global__ void GetNumTreeBit(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t num_vs,
    smask_t** sm_ptrs,
    const uint8_t ei
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (auto i = tid; i < res_size; i += num_threads)
    {
        // Build path_mask from all previously matched vertices
#if USE_CUM_PATH_MASK
        smask_t path_mask = SMASK_ALL;
#endif
        for (uint8_t k = 0; k < C_QV_COUNT - num_vs; k++) {
            uint8_t qv_k = C_ORDERS[oi].vs_[k];
            uint32_t vk = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - num_vs) + k) % C_RES_QUEUE.capability_];
#if USE_CUM_PATH_MASK
            path_mask &= sm_ptrs[ei * MAX_VCOUNT + qv_k][vk];
#endif
        }

        max_num_matches[i] = 1ul;
        for (uint8_t j = 0u; j < num_vs; j++)
        {
            const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - num_vs + j]];
            const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j]];
            const uint32_t& v = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - num_vs) + pre_qv_idx) % C_RES_QUEUE.capability_];
            const uint8_t target_qv = C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j];
            smask_t* sm_target = sm_ptrs[ei * MAX_VCOUNT + target_qv];

            uint32_t valid_count = 0;
            uint32_t size = index.sizes_[pre_qe_idx][v];
            for (uint32_t k = 0; k < size; k++) {
                uint32_t nbr = index.nbrs_[pre_qe_idx][v][k];
#if USE_CUM_PATH_MASK
                smask_t nbr_sm = sm_target[nbr];
                if (nbr_sm != SMASK_ZERO && (path_mask & nbr_sm) != SMASK_ZERO)
#else
                if (smask_read(sm_target, nbr))
#endif
                    valid_count++;
            }
            max_num_matches[i] *= valid_count;
        }
    }
}

// ============== Support Mask Variants ==============

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
) {
    __shared__ uint32_t visited[NWARP_PER_BLOCK][WARP_SIZE][MAX_VCOUNT - 1];
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    unsigned long long res_index;
    uint32_t map_v;
    unsigned long res_index_index;
    for (auto i = gwarp_id; i < DIV_CEIL(*total_i, WARP_SIZE); i += num_warps)
    {
        bool found = true;
        const unsigned long ii = i * WARP_SIZE + lane_id;
        if (ii >= *total_i) found = false;
        if (found)
        {
            res_index = lower_bound(max_num_matches + 1, res_size, ii + 1);

            res_index_index = ii - max_num_matches[res_index];
            for (uint8_t k = 0u; k < C_QV_COUNT - num_vs; k++)
            {
                visited[warp_id][lane_id][k] = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs) + k) % C_RES_QUEUE.capability_];
            }
        }
        __syncwarp();
        // Build initial path_mask from all vertices in the partial result
#if USE_CUM_PATH_MASK
        smask_t path_mask = SMASK_ALL;
#endif
        if (found) {
            for (uint8_t k = 0u; k < C_QV_COUNT - num_vs; k++) {
                uint8_t qv_k = C_ORDERS[oi].vs_[k];
#if USE_CUM_PATH_MASK
                path_mask &= sm_ptrs[ei * MAX_VCOUNT + qv_k][visited[warp_id][lane_id][k]];
#endif
            }
#if USE_CUM_PATH_MASK
            if (path_mask == SMASK_ZERO) found = false;
#endif
        }
        for (uint8_t j = 0u; j < num_vs; j++)
        {
            if (found)
            {
                const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[C_QV_COUNT - num_vs + j]];
                const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j]];
                const uint32_t& v = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs) + pre_qv_idx) % C_RES_QUEUE.capability_];

                map_v = index.nbrs_[pre_qe_idx][v][res_index_index % index.sizes_[pre_qe_idx][v]];
                res_index_index /= index.sizes_[pre_qe_idx][v];

                // Support mask check with path_mask filtering
                uint8_t target_qv = C_ORDERS[oi].vs_[C_QV_COUNT - num_vs + j];
#if USE_CUM_PATH_MASK
                smask_t map_v_sm = sm_ptrs[ei * MAX_VCOUNT + target_qv][map_v];
                if (map_v_sm == SMASK_ZERO)
                {
                    found = false;
                }
                else
                {
                    path_mask &= map_v_sm;
                    if (path_mask == SMASK_ZERO) found = false;
                }
#else
                if (!smask_read(sm_ptrs[ei * MAX_VCOUNT + target_qv], map_v))
                {
                    found = false;
                }
#endif

                for (uint8_t k = 0u; k < C_QV_COUNT - num_vs + j; k++)
                {
                    if (map_v == visited[warp_id][lane_id][k])
                    {
                        found = false;
                        break;
                    }
                }
                if (!found) break;
                if (j != num_vs - 1) visited[warp_id][lane_id][C_QV_COUNT - num_vs + j] = map_v;
            }
        }
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0)
        {
            atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}

#ifdef USE_MERGED_MATCHING
__global__ void GetNumTreeAllBit(
    const unsigned long long res,
    const unsigned long long res_size,
    unsigned long *max_num_matches,
    const RelationsGPU* d_all_local,
    const RelationsGPU update_index,
    smask_t** edge_sm_ptrs,
    const uint8_t num_vs
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = gridDim.x * blockDim.x;

    for (auto i = tid; i < res_size; i += num_threads)
    {
        uint32_t packed_v0 = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - num_vs)) % C_RES_QUEUE.capability_];
        uint8_t ei = packed_v0 >> 27;

        max_num_matches[i] = 1ul;
        for (uint8_t j = 0u; j < num_vs; j++)
        {
            const uint8_t& pre_qv_idx = C_GLOBAL_ORDER.bni_[C_GLOBAL_ORDER.bni_offs_[C_QV_COUNT - num_vs + j]];
            const uint8_t& pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[pre_qv_idx] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[C_QV_COUNT - num_vs + j]];

            uint32_t v;
            if (pre_qv_idx == 0) {
                v = packed_v0 & 0x07FFFFFF;
            } else {
                v = C_RES_QUEUE.array_[(res + i * (C_QV_COUNT - num_vs) + pre_qv_idx) % C_RES_QUEUE.capability_];
            }

            const uint8_t target_qv = C_GLOBAL_ORDER.vs_[C_QV_COUNT - num_vs + j];

            uint32_t base_size = d_all_local[ei].sizes_[pre_qe_idx][v];
            bool up_vis = (C_DIR_TO_EDGE[pre_qe_idx] < ei);
            uint32_t update_size = up_vis ? update_index.sizes_[pre_qe_idx][v] : 0;
            max_num_matches[i] *= (base_size + update_size);
        }
    }
}

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
) {
    __shared__ uint32_t visited[NWARP_PER_BLOCK][WARP_SIZE][MAX_VCOUNT - 1];
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    unsigned long long res_index;
    uint32_t map_v;
    unsigned long res_index_index;
    for (auto i = gwarp_id; i < DIV_CEIL(*total_i, WARP_SIZE); i += num_warps)
    {
        bool found = true;
        const unsigned long ii = i * WARP_SIZE + lane_id;
        if (ii >= *total_i) found = false;

        // Initialize path_mask from v0 and v1 support masks
#if USE_CUM_PATH_MASK
        smask_t path_mask = SMASK_ALL;
#endif
        if (found)
        {
            res_index = lower_bound(max_num_matches + 1, res_size, ii + 1);
            res_index_index = ii - max_num_matches[res_index];

            uint32_t packed_v0 = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs)) % C_RES_QUEUE.capability_];
            uint32_t v0 = packed_v0 & 0x07FFFFFF;
            visited[warp_id][lane_id][0] = v0;
            for (uint8_t k = 1u; k < C_QV_COUNT - num_vs; k++)
            {
                visited[warp_id][lane_id][k] = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs) + k) % C_RES_QUEUE.capability_];
            }

#if USE_CUM_PATH_MASK
            // Build initial path_mask from all vertices in the partial result
            uint8_t ei = packed_v0 >> 27;
            for (uint8_t k = 0u; k < C_QV_COUNT - num_vs; k++)
            {
                uint8_t qv = C_GLOBAL_ORDER.vs_[k];
                uint32_t vk = visited[warp_id][lane_id][k];
                path_mask &= edge_sm_ptrs[ei * MAX_VCOUNT + qv][vk];
            }
            if (path_mask == SMASK_ZERO) found = false;
#endif
        }
        __syncwarp();

        uint8_t ei = 0;
        if (found) {
            uint32_t p0 = C_RES_QUEUE.array_[(res + res_index * (C_QV_COUNT - num_vs)) % C_RES_QUEUE.capability_];
            ei = p0 >> 27;
        }

        for (uint8_t j = 0u; j < num_vs; j++)
        {
            if (found)
            {
                const uint8_t& pre_qv_idx = C_GLOBAL_ORDER.bni_[C_GLOBAL_ORDER.bni_offs_[C_QV_COUNT - num_vs + j]];
                const uint8_t& pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[pre_qv_idx] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[C_QV_COUNT - num_vs + j]];
                const uint32_t& v = visited[warp_id][lane_id][pre_qv_idx];

                uint32_t base_size = d_all_local[ei].sizes_[pre_qe_idx][v];
                bool up_vis = (C_DIR_TO_EDGE[pre_qe_idx] < ei);
                uint32_t update_size = up_vis ? update_index.sizes_[pre_qe_idx][v] : 0;
                uint32_t total_size = base_size + update_size;

                uint32_t local_idx = res_index_index % total_size;
                res_index_index /= total_size;

                if (local_idx < base_size)
                    map_v = d_all_local[ei].nbrs_[pre_qe_idx][v][local_idx];
                else
                    map_v = update_index.nbrs_[pre_qe_idx][v][local_idx - base_size];

                // Per-edge support mask check for target query vertex + path_mask filtering
                uint8_t target_qv = C_GLOBAL_ORDER.vs_[C_QV_COUNT - num_vs + j];
#if USE_CUM_PATH_MASK
                smask_t map_v_sm = edge_sm_ptrs[ei * MAX_VCOUNT + target_qv][map_v];
                if (map_v_sm == SMASK_ZERO)
                {
                    found = false;
                }
                else
                {
                    path_mask &= map_v_sm;
                    if (path_mask == SMASK_ZERO) found = false;
                }
#else
                if (!smask_read(edge_sm_ptrs[ei * MAX_VCOUNT + target_qv], map_v))
                {
                    found = false;
                }
#endif

                for (uint8_t k = 0u; k < C_QV_COUNT - num_vs + j; k++)
                {
                    if (map_v == visited[warp_id][lane_id][k])
                    {
                        found = false;
                        break;
                    }
                }
                if (!found) break;
                if (j != num_vs - 1) visited[warp_id][lane_id][C_QV_COUNT - num_vs + j] = map_v;
            }
        }
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        if (lane_id == 0)
        {
            atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}
#endif
