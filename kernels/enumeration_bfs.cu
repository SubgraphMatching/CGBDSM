#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/graph_gpu.h"


__global__ void extendBFSRegTwo(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const bool write_res
) {
    __shared__ uint32_t partial[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t nbr_size[NWARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    if (gwarp_id >= res_size)
    {
        return;
    }

    uint32_t v0 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capability_];
    uint32_t v1 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capability_];
    if (lane_id < start_depth - 2)
    {
        partial[warp_id][lane_id] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 2 + lane_id) % C_RES_QUEUE.capability_];
    }
    __syncwarp();
#ifdef DEBUG
    if(lane_id == 0)
    {
        printf("%d %d\n", v0, v1);
    }
    __syncwarp();
#endif

    const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[start_depth]];
    const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[start_depth]];

    if (lane_id == 0)
    {
        if (pre_qv_idx == 0u)
        {
            nbr_size[warp_id] = index.sizes_[pre_qe_idx][v0];
        }
        else if (pre_qv_idx == 1u)
        {
            nbr_size[warp_id] = index.sizes_[pre_qe_idx][v1];
        }
        else
        {
            nbr_size[warp_id] = index.sizes_[pre_qe_idx][partial[warp_id][pre_qv_idx - 2]];
        }
    }
    __syncwarp();

    for (uint32_t i = 0; i < DIV_CEIL(nbr_size[warp_id], WARP_SIZE); i++)
    {
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        uint32_t temp_nbr = UINT32_MAX;
        if (i * WARP_SIZE + lane_id < nbr_size[warp_id])
        {
            // 3. each lane gets a neighbor of a vertex in the previous level
            if (pre_qv_idx == 0)
            {
                temp_nbr = index.nbrs_[pre_qe_idx][v0][i * WARP_SIZE + lane_id];
            }
            else if (pre_qv_idx == 1)
            {
                temp_nbr = index.nbrs_[pre_qe_idx][v1][i * WARP_SIZE + lane_id];
            }
            else
            {
                temp_nbr = index.nbrs_[pre_qe_idx][
                    partial[warp_id][pre_qv_idx - 2]
                ][i * WARP_SIZE + lane_id];
            }
        }

        bool found = i * WARP_SIZE + lane_id < nbr_size[warp_id];
        __syncwarp();
        if (found)
        {
#ifdef DEBUG
            printf("found1 %d temp_nbr=%d\n", lane_id, temp_nbr);
#endif
            if (temp_nbr == v0 || temp_nbr == v1)
            {
                found = false;
            }
            else
            {
                for (uint8_t j = 2u; j < start_depth; j++)
                {
                    if (partial[warp_id][j - 2u] == temp_nbr)
                    {
                        found = false;
                        break;
                    }
                }
            }
        }
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        if (found)
        {
#ifdef DEBUG
            printf("found2 %d\n", lane_id);
#endif
            for (uint8_t off = C_ORDERS[oi].bni_offs_[start_depth] + 1; off < C_ORDERS[oi].bni_offs_[start_depth + 1]; off++)
            {
                const uint8_t& bni = C_ORDERS[oi].bni_[off];
                const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[start_depth]];
                const uint32_t& pre_pre_v = bni == 0 ? v0 : (bni == 1 ? v1 : partial[warp_id][bni - 2]);
#ifdef DEBUG
                printf("bni=%d, pre_pre_qe_idx=%d, pre_pre_v=%d\n", bni, pre_pre_qe_idx, pre_pre_v);
#endif
                const uint32_t res = lower_bound(index.nbrs_[pre_pre_qe_idx][pre_pre_v], index.sizes_[pre_pre_qe_idx][pre_pre_v], temp_nbr);
                if (res == index.sizes_[pre_pre_qe_idx][pre_pre_v] || index.nbrs_[pre_pre_qe_idx][pre_pre_v][res] != temp_nbr)
                {
                    found = false;
                    break;
                }
            }
        }
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;

        // 6. write the local candidates to result_queue and their group id to group_id
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
        
        if (write_res)
        {
            if (found_mask)
            {
                unsigned long long int write_pos;
                if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                write_pos = __shfl_sync(0xffffffff, write_pos, 0, 64);
                if (write_pos + __popc(found_mask) > h_max_new_res_size_) return;
                if (found)
                {
                    write_pos += rank;
                    C_RES_QUEUE.array_[(new_res + write_pos * (start_depth + 1)) % C_RES_QUEUE.capability_] = v0;
                    C_RES_QUEUE.array_[(new_res + write_pos * (start_depth + 1) + 1) % C_RES_QUEUE.capability_] = v1;

                    for (uint8_t j = 2u; j < start_depth; j++)
                    {
                        C_RES_QUEUE.array_[(new_res + write_pos * (start_depth + 1) + j) % C_RES_QUEUE.capability_] = partial[warp_id][j - 2];
                    }
                    C_RES_QUEUE.array_[(new_res + write_pos * (start_depth + 1) + start_depth) % C_RES_QUEUE.capability_] = temp_nbr;
                }
            }
        }
        else
        {
            if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}

__global__ void extendBFSShareAll(
    const unsigned long long res,
    const unsigned long long int res_size,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    const RelationsGPU index,
    const uint8_t oi,
    const uint8_t start_depth,
    const bool write_res
) {
    __shared__ uint32_t partial[NWARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint32_t nbr_size[NWARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;

    if (gwarp_id >= res_size)
    {
        return;
    }

    if (lane_id < start_depth)
    {
        partial[warp_id][lane_id] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + lane_id) % C_RES_QUEUE.capability_];
    }
    __syncwarp();
#ifdef DEBUG
    if(lane_id == 0)
    {
        printf("%d %d\n", v0, v1);
    }
    __syncwarp();
#endif

    const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[start_depth]];
    const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[start_depth]];

    if (lane_id == 0)
    {
        nbr_size[warp_id] = index.sizes_[pre_qe_idx][partial[warp_id][pre_qv_idx]];
    }
    __syncwarp();

    for (uint32_t i = 0; i < DIV_CEIL(nbr_size[warp_id], WARP_SIZE); i++)
    {
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        uint32_t temp_nbr = UINT32_MAX;
        if (i * WARP_SIZE + lane_id < nbr_size[warp_id])
        {
            temp_nbr = index.nbrs_[pre_qe_idx][partial[warp_id][pre_qv_idx]][i * WARP_SIZE + lane_id];
        }

        bool found = i * WARP_SIZE + lane_id < nbr_size[warp_id];
        __syncwarp();
        if (found)
        {
#ifdef DEBUG
            printf("found1 %d temp_nbr=%d\n", lane_id, temp_nbr);
#endif
            for (uint8_t j = 0u; j < start_depth; j++)
            {
                if (partial[warp_id][j] == temp_nbr)
                {
                    found = false;
                    break;
                }
            }
        }
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        if (found)
        {
#ifdef DEBUG
            printf("found2 %d\n", lane_id);
#endif
            for (uint8_t off = C_ORDERS[oi].bni_offs_[start_depth] + 1; off < C_ORDERS[oi].bni_offs_[start_depth + 1]; off++)
            {
                const uint8_t& bni = C_ORDERS[oi].bni_[off];
                const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[start_depth]];
                const uint32_t& pre_pre_v = partial[warp_id][bni];
#ifdef DEBUG
                printf("bni=%d, pre_pre_qe_idx=%d, pre_pre_v=%d\n", bni, pre_pre_qe_idx, pre_pre_v);
#endif
                const uint32_t res = lower_bound(index.nbrs_[pre_pre_qe_idx][pre_pre_v], index.sizes_[pre_pre_qe_idx][pre_pre_v], temp_nbr);
                if (res == index.sizes_[pre_pre_qe_idx][pre_pre_v] || index.nbrs_[pre_pre_qe_idx][pre_pre_v][res] != temp_nbr)
                {
                    found = false;
                    break;
                }
            }
        }
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;

        // 6. write the local candidates to result_queue and their group id to group_id
        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
        const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
        
        if (write_res)
        {
            if (found_mask)
            {
                unsigned long long int write_pos;
                if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                write_pos = __shfl_sync(0xffffffff, write_pos, 0, 64);
                if (write_pos + __popc(found_mask) > h_max_new_res_size_) return;
                if (found)
                {
                    write_pos += rank;
                    for (uint8_t j = 0u; j < start_depth; j++)
                    {
                        C_RES_QUEUE.array_[(new_res + write_pos * (start_depth + 1) + j) % C_RES_QUEUE.capability_] = partial[warp_id][j];
                    }
                    C_RES_QUEUE.array_[(new_res + write_pos * (start_depth + 1) + start_depth) % C_RES_QUEUE.capability_] = temp_nbr;
                }
            }
        }
        else
        {
            if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
        }
        __syncwarp();
    }
}
