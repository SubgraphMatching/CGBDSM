#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/graph_gpu.h"

__global__ void write_initial_partial_results(
    RelationsGPU index_gpu,
    const uint8_t idx,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
) {
    __shared__ unsigned long long int write_pos[NWARP_PER_BLOCK];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t v = gwarp_id; v < C_DV_COUNT; v+= num_warps)
    {
        if (lane_id == 0) write_pos[warp_id] = atomicAdd(new_res_size, (unsigned long long)index_gpu.sizes_[idx][v]);
        __syncwarp();
        //if (*new_res_size >= h_max_new_res_size) return;
        for (uint32_t j = lane_id; j < index_gpu.sizes_[idx][v]; j += WARP_SIZE)
        {
            C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + j) * 2] = v;
            C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + j) * 2 + 1] = index_gpu.nbrs_[idx][v][j];
        }
    }
}

__global__ void write_initial_partial_results_bit(
    RelationsGPU index_gpu,
    const uint8_t idx,
    const uint8_t qv0,
    const uint8_t qv1,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
) {
    __shared__ unsigned long long int write_pos[NWARP_PER_BLOCK];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t v = gwarp_id; v < C_DV_COUNT; v += num_warps)
    {
        if ((C_VALID_BITS.bits_[qv0][v / 32u] & (1u << (v % 32u))) == 0u) continue;
        uint32_t size = index_gpu.sizes_[idx][v], my_count = 0;
        for (uint32_t j = lane_id; j < size; j += WARP_SIZE) {
            uint32_t nbr = index_gpu.nbrs_[idx][v][j];
            if ((C_VALID_BITS.bits_[qv1][nbr / 32u] & (1u << (nbr % 32u))) != 0u) 
                my_count++;
        }

        uint32_t inclusive = my_count;
        for (uint32_t d = 1; d < WARP_SIZE; d *= 2) {
            uint32_t n = __shfl_up_sync(0xffffffff, inclusive, d);
            if (lane_id >= d) inclusive += n;
        }
        uint32_t lane_offset = inclusive - my_count;
        uint32_t total = __shfl_sync(0xffffffff, inclusive, WARP_SIZE - 1);
        if (lane_id == 0) write_pos[warp_id] = atomicAdd(new_res_size, (unsigned long long)total);
        __syncwarp();

        uint32_t local_pos = lane_offset;
        for (uint32_t j = lane_id; j < size; j += WARP_SIZE) {
            uint32_t nbr = index_gpu.nbrs_[idx][v][j];
            if ((C_VALID_BITS.bits_[qv1][nbr / 32u] & (1u << (nbr % 32u))) != 0u) {
                C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + local_pos) * 2] = v;
                C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + local_pos) * 2 + 1] = nbr;
                local_pos++;
            }
        }
    }
}

__global__ void write_cpu_partial_results(
    const uint32_t* cpu_results,
    const uint32_t num_cpu_results,
    const uint32_t start_depth,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
) {
    __shared__ unsigned long long int write_pos[NWARP_PER_BLOCK];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t r = gwarp_id; r < num_cpu_results; r += num_warps)
    {
        if (lane_id == 0) {
            unsigned long long int pos = atomicAdd(new_res_size, 1ULL);
            if (pos >= h_max_new_res_size) {
                write_pos[warp_id] = UINT64_MAX;  // 表示超出范围
            } else {
                write_pos[warp_id] = pos;
            }
        }
        __syncwarp();
        if (write_pos[warp_id] == UINT64_MAX) continue;
        // 每个warp写入一个部分匹配结果
        for (uint32_t j = lane_id; j < start_depth; j += WARP_SIZE)
        {
            unsigned long long int idx = (new_res + write_pos[warp_id] * start_depth + j) % C_RES_QUEUE.capability_;
            C_RES_QUEUE.array_[idx] = cpu_results[r * start_depth + j];
        }
    }
}

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
) {
    __shared__ uint32_t result_queue[NWARP_PER_BLOCK][MAX_VCOUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ uint8_t end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ bool intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ uint8_t rem_nbr_count[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t depth[NWARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    if (gwarp_id >= res_size)
        return;
    uint32_t v0 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capability_];
    uint32_t v1 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capability_];
    if (lane_id < start_depth - 2)
        result_queue[warp_id][lane_id][0] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 2 + lane_id) % C_RES_QUEUE.capability_];
    __syncwarp();

    if (lane_id < C_QV_COUNT - 2)
    {
        queue_pos[warp_id][lane_id] = 0u;
        queue_size[warp_id][lane_id] = (lane_id < start_depth - 2) ? 1u: 0u;
        end_v[warp_id][lane_id] = 0u;
        end_nbr[warp_id][lane_id] = 0u;
        intersection_continue[warp_id][lane_id] = false;
    }
    if (lane_id == 0)
        depth[warp_id] = start_depth;
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
        if (queue_pos[warp_id][depth[warp_id] - 2] >= queue_size[warp_id][depth[warp_id] - 2])
        {
            if (
                intersection_continue[warp_id][depth[warp_id] - 2] && 
                ((pre_qv_idx < 2 && end_v[warp_id][depth[warp_id] - 2] > 0) ||
                (pre_qv_idx >= 2 && end_v[warp_id][depth[warp_id] - 2] > queue_pos[warp_id][pre_qv_idx - 2]))
            ) {
                // no more intersection to do, go back to the previous level
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;

                    intersection_continue[warp_id][depth[warp_id] - 2] = false;

                    depth[warp_id]--;

                    if (depth[warp_id] >= 2)
                        queue_pos[warp_id][depth[warp_id] - 2]++;
                }
                __syncwarp();
            }
            else
            {
                if (!intersection_continue[warp_id][depth[warp_id] - 2])
                {
                    if (lane_id == 0)
                    {
                        if (pre_qv_idx < 2)
                        {
                            end_v[warp_id][depth[warp_id] - 2] = 0u;
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                        else
                        {
                            end_v[warp_id][depth[warp_id] - 2] = queue_pos[warp_id][pre_qv_idx - 2];
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                    }
                    __syncwarp();
                }
                // need to start/continue the intersection
                
                intersection_continue[warp_id][depth[warp_id] - 2] = true;

                const uint8_t rem_pre_dv_count = 1u;
                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? v0 : UINT32_MAX;
                }
                else if (pre_qv_idx == 1u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? v1 : UINT32_MAX;
                }
                else
                {
                    pre_dv = lane_id < rem_pre_dv_count ? result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]] : UINT32_MAX;
                }

                // 2. retrieve the number of neighbors of each vertex
                // end_v[warp_id][depth[warp_id] - 2] should be equal to queue_pos[warp_id][pre_qv_idx - 2]
                rem_nbr_count[warp_id][lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id][depth[warp_id] - 2] : 0u), 64u)
                    : 0u;
                __syncwarp();
                if (lane_id > 0) rem_nbr_count[warp_id][lane_id] = rem_nbr_count[warp_id][0];
                __syncwarp();
                // each lane found a vertex temp_nbr
                uint32_t temp_nbr = UINT32_MAX;
                if (lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1])
                {
                    // 3. each lane gets a neighbor of a vertex in the previous level
                    if (pre_qv_idx == 0)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][v0][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else if (pre_qv_idx == 1)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][v1][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][
                            result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]]
                        ][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // update start_v/nbr and end_v/nbr by the active lane with the greatest id
                uint8_t num_active_lanes = __popc(__ballot_sync(0xffffffff, lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1]));
                if (num_active_lanes == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id] - 2] += rem_pre_dv_count;
                        end_nbr[warp_id][depth[warp_id] - 2] = 0;
                    }
                    __syncwarp();
                }
                else
                {
                    if (lane_id == num_active_lanes - 1)
                    {
                        if (rem_nbr_count[warp_id][0] > lane_id + 1)
                        {
                            // not all neighbors of the last vertex are processed
                            end_nbr[warp_id][depth[warp_id] - 2] += lane_id + 1;
                        }
                        else
                        {
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            end_v[warp_id][depth[warp_id] - 2] += 1u;
                        }
                    }
                    __syncwarp();
                }

                // 5. the lane search for the vertex on other nbr arrays
                bool found = lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1];
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
                    if (temp_nbr == v0 || temp_nbr == v1)
                    {
                        found = false;
                    }
                    else
                    {
                        for (uint8_t i = 2u; i < depth[warp_id]; i++)
                        {
                            if (result_queue[warp_id][i - 2u][queue_pos[warp_id][i - 2]] == temp_nbr)
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
                    for (uint8_t off = C_ORDERS[oi].bni_offs_[depth[warp_id]] + 1; off < C_ORDERS[oi].bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        const uint8_t& bni = C_ORDERS[oi].bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = bni == 0 ? v0 : (bni == 1 ? v1 : result_queue[warp_id][bni - 2][queue_pos[warp_id][bni - 2]]);
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
                if (depth[warp_id] < end_depth - 1)
                {
                    if (found) result_queue[warp_id][depth[warp_id] - 2][rank] = temp_nbr;
                }
                else
                {
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
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capability_] = v0;
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capability_] = v1;

                                for (uint8_t j = 2u; j < end_depth - 1; j++)
                                {
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capability_] = result_queue[warp_id][j - 2][queue_pos[warp_id][j - 2]];
                                }
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capability_] = temp_nbr;
                            }
                        }
                    }
                    else
                    {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                __syncwarp();
                if (found && rank == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = (depth[warp_id] < end_depth - 1) ? 0u : __popc(found_mask);
                    queue_size[warp_id][depth[warp_id] - 2] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else // go to the next level
        {
            if (lane_id == 0 && depth[warp_id] < end_depth - 1)
                depth[warp_id] ++;
            __syncwarp();
        }
    }
}

// Valid bit variant of extendBFSDFSRegTwo
// Identical except for the valid bit check after reading temp_nbr at the first BN iteration
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
    const bool write_res
) {
    __shared__ uint32_t result_queue[NWARP_PER_BLOCK][MAX_VCOUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ uint8_t end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ bool intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ uint8_t rem_nbr_count[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t depth[NWARP_PER_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    if (gwarp_id >= res_size)
        return;
    uint32_t v0 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capability_];
    uint32_t v1 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capability_];
    if (lane_id < start_depth - 2)
        result_queue[warp_id][lane_id][0] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 2 + lane_id) % C_RES_QUEUE.capability_];
    __syncwarp();

    if (lane_id < C_QV_COUNT - 2)
    {
        queue_pos[warp_id][lane_id] = 0u;
        queue_size[warp_id][lane_id] = (lane_id < start_depth - 2) ? 1u: 0u;
        end_v[warp_id][lane_id] = 0u;
        end_nbr[warp_id][lane_id] = 0u;
        intersection_continue[warp_id][lane_id] = false;
    }
    if (lane_id == 0)
        depth[warp_id] = start_depth;
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;
        const uint8_t& pre_qv_idx = C_ORDERS[oi].bni_[C_ORDERS[oi].bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[pre_qv_idx] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
        if (queue_pos[warp_id][depth[warp_id] - 2] >= queue_size[warp_id][depth[warp_id] - 2])
        {
            if (
                intersection_continue[warp_id][depth[warp_id] - 2] &&
                ((pre_qv_idx < 2 && end_v[warp_id][depth[warp_id] - 2] > 0) ||
                (pre_qv_idx >= 2 && end_v[warp_id][depth[warp_id] - 2] > queue_pos[warp_id][pre_qv_idx - 2]))
            ) {
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;

                    intersection_continue[warp_id][depth[warp_id] - 2] = false;

                    depth[warp_id]--;

                    if (depth[warp_id] >= 2)
                        queue_pos[warp_id][depth[warp_id] - 2]++;
                }
                __syncwarp();
            }
            else
            {
                if (!intersection_continue[warp_id][depth[warp_id] - 2])
                {
                    if (lane_id == 0)
                    {
                        if (pre_qv_idx < 2)
                        {
                            end_v[warp_id][depth[warp_id] - 2] = 0u;
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                        else
                        {
                            end_v[warp_id][depth[warp_id] - 2] = queue_pos[warp_id][pre_qv_idx - 2];
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                        }
                    }
                    __syncwarp();
                }

                intersection_continue[warp_id][depth[warp_id] - 2] = true;

                const uint8_t rem_pre_dv_count = 1u;
                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? v0 : UINT32_MAX;
                }
                else if (pre_qv_idx == 1u)
                {
                    pre_dv = lane_id < rem_pre_dv_count ? v1 : UINT32_MAX;
                }
                else
                {
                    pre_dv = lane_id < rem_pre_dv_count ? result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]] : UINT32_MAX;
                }

                rem_nbr_count[warp_id][lane_id] = lane_id < rem_pre_dv_count
                    ? min(index.sizes_[pre_qe_idx][pre_dv] - (lane_id == 0 ? end_nbr[warp_id][depth[warp_id] - 2] : 0u), 64u)
                    : 0u;
                __syncwarp();
                if (lane_id > 0) rem_nbr_count[warp_id][lane_id] = rem_nbr_count[warp_id][0];
                __syncwarp();

                uint32_t temp_nbr = UINT32_MAX;
                if (lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1])
                {
                    if (pre_qv_idx == 0)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][v0][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else if (pre_qv_idx == 1)
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][v1][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                    else
                    {
                        temp_nbr = index.nbrs_[pre_qe_idx][
                            result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]]
                        ][end_nbr[warp_id][depth[warp_id] - 2] + lane_id];
                    }
                }
                __syncwarp();

                // === Valid bit check (the ONLY difference from extendBFSDFSRegTwo) ===
                if (temp_nbr != UINT32_MAX) {
                    uint8_t current_qv = C_ORDERS[oi].vs_[depth[warp_id]];
                    if ((C_VALID_BITS.bits_[current_qv][temp_nbr / 32u] & (1u << (temp_nbr % 32u))) == 0u) {
                        temp_nbr = UINT32_MAX;
                    }
                }
                // === End valid bit check ===

                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                uint8_t num_active_lanes = __popc(__ballot_sync(0xffffffff, lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1]));
                if (num_active_lanes == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id] - 2] += rem_pre_dv_count;
                        end_nbr[warp_id][depth[warp_id] - 2] = 0;
                    }
                    __syncwarp();
                }
                else
                {
                    if (lane_id == num_active_lanes - 1)
                    {
                        if (rem_nbr_count[warp_id][0] > lane_id + 1)
                        {
                            end_nbr[warp_id][depth[warp_id] - 2] += lane_id + 1;
                        }
                        else
                        {
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            end_v[warp_id][depth[warp_id] - 2] += 1u;
                        }
                    }
                    __syncwarp();
                }

                bool found = lane_id < rem_nbr_count[warp_id][WARP_SIZE - 1];
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) return;
                if (found)
                {
                    if (temp_nbr == v0 || temp_nbr == v1)
                    {
                        found = false;
                    }
                    else
                    {
                        for (uint8_t i = 2u; i < depth[warp_id]; i++)
                        {
                            if (result_queue[warp_id][i - 2u][queue_pos[warp_id][i - 2]] == temp_nbr)
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
                    for (uint8_t off = C_ORDERS[oi].bni_offs_[depth[warp_id]] + 1; off < C_ORDERS[oi].bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        const uint8_t& bni = C_ORDERS[oi].bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_ORDERS[oi].vs_[bni] * C_QV_COUNT + C_ORDERS[oi].vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = bni == 0 ? v0 : (bni == 1 ? v1 : result_queue[warp_id][bni - 2][queue_pos[warp_id][bni - 2]]);
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

                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                if (depth[warp_id] < end_depth - 1)
                {
                    if (found) result_queue[warp_id][depth[warp_id] - 2][rank] = temp_nbr;
                }
                else
                {
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
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capability_] = v0;
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capability_] = v1;

                                for (uint8_t j = 2u; j < end_depth - 1; j++)
                                {
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capability_] = result_queue[warp_id][j - 2][queue_pos[warp_id][j - 2]];
                                }
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capability_] = temp_nbr;
                            }
                        }
                    }
                    else
                    {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                __syncwarp();
                if (found && rank == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = (depth[warp_id] < end_depth - 1) ? 0u : __popc(found_mask);
                    queue_size[warp_id][depth[warp_id] - 2] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else
        {
            if (lane_id == 0 && depth[warp_id] < end_depth - 1)
                depth[warp_id] ++;
            __syncwarp();
        }
    }
}
