#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/graph_gpu.h"

// // Device-global per-data-vertex conflict-free buffer (compressed ID space): 1 => candidate is
// // conflict-free (single-owned), relax its u0/u1 connection check (mask guarantees the adjacency).
// // nullptr/all-zero => baseline (no relaxation). Set per batch by UploadConflictFree.
// __device__ const uint8_t* g_d_cf = nullptr;

// #if RELAX_DEBUG
// // Debug: count relax-pass cases where the skipped binary search would have FAILED.
// // Whole-process cumulative (zero-initialized device global, no per-batch reset).
// __device__ uint64_t g_dbg_skip_fail = 0;
// #endif

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
    const unsigned long long int h_max_new_res_size,
    smask_t** sm_ptrs,
    const uint8_t ei
) {
    __shared__ unsigned long long int write_pos[NWARP_PER_BLOCK];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    smask_t* sm_qv0 = sm_ptrs[ei * MAX_VCOUNT + qv0];
    smask_t* sm_qv1 = sm_ptrs[ei * MAX_VCOUNT + qv1];

    for (uint32_t v = gwarp_id; v < C_DV_COUNT; v += num_warps)
    {
        auto v_sm = smask_read(sm_qv0, v);
        if (!v_sm) continue;
        uint32_t size = index_gpu.sizes_[idx][v], my_count = 0;
        for (uint32_t j = lane_id; j < size; j += WARP_SIZE) {
            uint32_t nbr = index_gpu.nbrs_[idx][v][j];
            if (smask_pair_ok(v_sm, sm_qv1, nbr))
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
            if (smask_pair_ok(v_sm, sm_qv1, nbr)) {
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

// Support mask variant of extendBFSDFSRegTwo
// Uses smask_t support masks with cumulative path mask filtering instead of 1-bit ValidBits
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
) {
    __shared__ uint32_t result_queue[NWARP_PER_BLOCK][MAX_VCOUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ uint8_t end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ bool intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    __shared__ uint8_t depth[NWARP_PER_BLOCK];

    __shared__ uint32_t compact_nbrs[NWARP_PER_BLOCK][WARP_SIZE * 2];
    __shared__ uint32_t compact_pos[NWARP_PER_BLOCK][WARP_SIZE * 2];
    __shared__ uint8_t compact_count[NWARP_PER_BLOCK];

#if USE_CUM_PATH_MASK
    __shared__ smask_t cum_path_mask[NWARP_PER_BLOCK][MAX_VCOUNT]; // cached cumulative path mask per depth
#endif

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
        compact_count[warp_id] = 0;
    }
    if (lane_id == 0) {
        depth[warp_id] = start_depth;
#if USE_CUM_PATH_MASK
        // Initialize cumulative path mask cache
        cum_path_mask[warp_id][0] = sm_ptrs[ei * MAX_VCOUNT + C_ORDERS[oi].vs_[0]][v0];
        cum_path_mask[warp_id][1] = cum_path_mask[warp_id][0]
            & sm_ptrs[ei * MAX_VCOUNT + C_ORDERS[oi].vs_[1]][v1];
        for (uint8_t d = 2; d < start_depth; d++) {
            uint32_t vi = result_queue[warp_id][d - 2][0];
            cum_path_mask[warp_id][d] = cum_path_mask[warp_id][d - 1]
                & sm_ptrs[ei * MAX_VCOUNT + C_ORDERS[oi].vs_[d]][vi];
        }
#endif
    }
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
                            compact_count[warp_id] = 0;
                        }
                        else
                        {
                            end_v[warp_id][depth[warp_id] - 2] = queue_pos[warp_id][pre_qv_idx - 2];
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            compact_count[warp_id] = 0;
                        }
                    }
                    __syncwarp();
                }

                intersection_continue[warp_id][depth[warp_id] - 2] = true;

                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u)
                    pre_dv = v0;
                else if (pre_qv_idx == 1u)
                    pre_dv = v1;
                else
                    pre_dv = result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]];

                const uint32_t read_end = index.sizes_[pre_qe_idx][pre_dv];
                const uint8_t current_qv = C_ORDERS[oi].vs_[depth[warp_id]];
#if USE_CUM_PATH_MASK
                // Read cumulative path mask from cache (O(1) shared memory read)
                smask_t parent_pm = cum_path_mask[warp_id][depth[warp_id] - 1];
#endif
                smask_t* const cur_vb = sm_ptrs[ei * MAX_VCOUNT + current_qv];
                uint32_t read_offset = end_nbr[warp_id][depth[warp_id] - 2];
                if (lane_id == 0) compact_count[warp_id] = 0;
                __syncwarp();

                while (compact_count[warp_id] < WARP_SIZE && read_offset < read_end)
                {
                    uint32_t nbr = UINT32_MAX;
                    if (read_offset + lane_id < read_end)
                    {
                        if (pre_qv_idx == 0)
                            nbr = index.nbrs_[pre_qe_idx][v0][read_offset + lane_id];
                        else if (pre_qv_idx == 1)
                            nbr = index.nbrs_[pre_qe_idx][v1][read_offset + lane_id];
                        else
                            nbr = index.nbrs_[pre_qe_idx][
                                result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]]
                            ][read_offset + lane_id];

#if USE_CUM_PATH_MASK
                        if (cur_vb[nbr] == SMASK_ZERO || (parent_pm & cur_vb[nbr]) == SMASK_ZERO)
#else
                        if (!smask_read(cur_vb, nbr))
#endif
                            nbr = UINT32_MAX;
                    }
                    __syncwarp();

                    uint32_t ballot = __ballot_sync(0xffffffff, nbr != UINT32_MAX);
                    uint8_t num_new = __popc(ballot);
                    uint8_t my_rank = __popc(ballot & ((1u << lane_id) - 1));

                    if (nbr != UINT32_MAX) {
                        compact_nbrs[warp_id][compact_count[warp_id] + my_rank] = nbr;
                        compact_pos[warp_id][compact_count[warp_id] + my_rank] = read_offset + lane_id;
                    }

                    if (lane_id == 0) compact_count[warp_id] += num_new;
                    read_offset += WARP_SIZE;
                    __syncwarp();
                }

                uint32_t num_valid = min(WARP_SIZE, compact_count[warp_id]);
                uint32_t temp_nbr = lane_id < num_valid
                    ? compact_nbrs[warp_id][lane_id]
                    : UINT32_MAX;
                uint32_t temp_pos = lane_id < num_valid
                    ? compact_pos[warp_id][lane_id]
                    : UINT32_MAX;

                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                if (num_valid == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id] - 2] += 1u;
                        end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                    }
                    __syncwarp();
                } else {
                    if (lane_id == num_valid - 1)
                    {
                        uint32_t total_check_end = min(temp_pos + 1, read_end);
                        if (total_check_end >= read_end) {
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            end_v[warp_id][depth[warp_id] - 2] += 1u;
                        } else {
                            end_nbr[warp_id][depth[warp_id] - 2] = total_check_end;
                        }
                    }
                    __syncwarp();
                }

                bool found = lane_id < num_valid;
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
            if (lane_id == 0 && depth[warp_id] < end_depth - 1) {
#if USE_CUM_PATH_MASK
                // Update cumulative path mask cache before going deeper
                uint8_t cur_d = depth[warp_id];
                uint32_t vi = result_queue[warp_id][cur_d - 2][queue_pos[warp_id][cur_d - 2]];
                cum_path_mask[warp_id][cur_d] = cum_path_mask[warp_id][cur_d - 1]
                    & sm_ptrs[ei * MAX_VCOUNT + C_ORDERS[oi].vs_[cur_d]][vi];
#endif
                depth[warp_id] ++;
            }
            __syncwarp();
        }
    }
}
#ifdef USE_MERGED_MATCHING
__global__ void writeInitialResultsAllBit(
    const RelationsGPU* d_all_local,
    const RelationsGPU d_update_index,
    smask_t** edge_sm_ptrs,
    const bool* edge_ok,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
) {
    const uint8_t ei = blockIdx.y;
    if (!edge_ok[ei]) return;

    __shared__ unsigned long long int write_pos[NWARP_PER_BLOCK];
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps_per_edge = gridDim.x * blockDim.x / WARP_SIZE;

    const uint8_t qv0 = C_GLOBAL_ORDER.vs_[0];
    const uint8_t qv1 = C_GLOBAL_ORDER.vs_[1];
    const uint8_t idx = C_EIDX[qv0 * C_QV_COUNT + qv1];

    // Per-edge support mask access: [matching_order][query_vertex]
    smask_t* sm_qv0 = edge_sm_ptrs[ei * MAX_VCOUNT + qv0];
    smask_t* sm_qv1 = edge_sm_ptrs[ei * MAX_VCOUNT + qv1];
    uint32_t** nbrs = d_all_local[ei].nbrs_[idx];
    uint32_t* sizes = d_all_local[ei].sizes_[idx];
    uint32_t** nbrs_update = d_update_index.nbrs_[idx];
    uint32_t* sizes_update = d_update_index.sizes_[idx];

    for (uint32_t v = gwarp_id; v < C_DV_COUNT; v += num_warps_per_edge)
    {
        auto v_sm = smask_read(sm_qv0, v);
        if (!v_sm) continue;
        uint32_t size = sizes[v], my_count = 0;
        for (uint32_t j = lane_id; j < size; j += WARP_SIZE) {
            uint32_t nbr = nbrs[v][j];
            if (smask_pair_ok(v_sm, sm_qv1, nbr))
                my_count++;
        }

        if((C_DIR_TO_EDGE[idx] < ei)) {
            uint32_t size_update = sizes_update[v];
            for (uint32_t j = lane_id; j < size_update; j += WARP_SIZE) {
                uint32_t nbr = nbrs_update[v][j];
                if (smask_pair_ok(v_sm, sm_qv1, nbr))
                    my_count++;
            }
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
            uint32_t nbr = nbrs[v][j];
            if (smask_pair_ok(v_sm, sm_qv1, nbr)) {
                C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + local_pos) * 2] = ((uint32_t)ei << 27) | v;
                C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + local_pos) * 2 + 1] = nbr;
                local_pos++;
            }
        }

        if((C_DIR_TO_EDGE[idx] < ei)) {
            uint32_t size_update = sizes_update[v];
            for (uint32_t j = lane_id; j < size_update; j += WARP_SIZE) {
                uint32_t nbr = nbrs_update[v][j];
                if (smask_pair_ok(v_sm, sm_qv1, nbr)) {
                    C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + local_pos) * 2] = ((uint32_t)ei << 27) | v;
                    C_RES_QUEUE.array_[new_res + (write_pos[warp_id] + local_pos) * 2 + 1] = nbr;
                    local_pos++;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(BLOCK_DIM, 2)
extendBFSAllBit(
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
) {
    __shared__ uint32_t result_queue[NWARP_PER_BLOCK][MAX_VCOUNT - 2][WARP_SIZE]; // 这个 result_queue 利用率怎么样？
    __shared__ uint8_t queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ bool intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t depth[NWARP_PER_BLOCK];
    __shared__ uint32_t compact_nbrs[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t compact_pos[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t compact_count[NWARP_PER_BLOCK];
#if USE_CUM_PATH_MASK
    __shared__ smask_t cum_path_mask[NWARP_PER_BLOCK][MAX_VCOUNT]; // cached cumulative path mask per depth
#endif

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long total_warps = ((unsigned long long)gridDim.x * blockDim.x) / WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long) blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    // const unsigned long long gwarp_id = (unsigned long long)warp_id * gridDim.x + blockIdx.x;
    if (gwarp_id >= res_size) return;

    uint32_t v0 = 0, v1 = 0;
    uint8_t ei = 0;
    uint32_t packed_v0 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capability_];
    ei = packed_v0 >> 27;
    v0 = packed_v0 & 0x07FFFFFF;
    v1 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capability_];

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
        compact_count[warp_id] = 0;
    }
    if (lane_id == 0) {
        depth[warp_id] = start_depth;
#if USE_CUM_PATH_MASK
        cum_path_mask[warp_id][0] = edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[0]][v0];
        cum_path_mask[warp_id][1] = cum_path_mask[warp_id][0] & edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[1]][v1];
        for (uint8_t i = 2; i < start_depth; i++) {
            uint32_t vi = result_queue[warp_id][i - 2][0];
            cum_path_mask[warp_id][i] = cum_path_mask[warp_id][i - 1] & edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[i]][vi];
        }
#endif
    }
    __syncwarp();

    while (depth[warp_id] >= start_depth)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_)
            return;
        const uint8_t& pre_qv_idx = C_GLOBAL_ORDER.bni_[C_GLOBAL_ORDER.bni_offs_[depth[warp_id]]];
        const uint8_t& pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[pre_qv_idx] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[depth[warp_id]]];
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
                            compact_count[warp_id] = 0;
                        }
                        else
                        {
                            end_v[warp_id][depth[warp_id] - 2] = queue_pos[warp_id][pre_qv_idx - 2];
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            compact_count[warp_id] = 0;
                        }
                    }
                    __syncwarp();
                }

                intersection_continue[warp_id][depth[warp_id] - 2] = true;

                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u)
                    pre_dv = v0;
                else if (pre_qv_idx == 1u)
                    pre_dv = v1;
                else
                    pre_dv = result_queue[warp_id][pre_qv_idx - 2][queue_pos[warp_id][pre_qv_idx - 2]];

                const uint32_t base_end = d_all_local[ei].sizes_[pre_qe_idx][pre_dv];
                const bool up_vis = (C_DIR_TO_EDGE[pre_qe_idx] < ei);
                const uint32_t update_sz = up_vis ? update_index.sizes_[pre_qe_idx][pre_dv] : 0;
                const uint32_t read_end = base_end + update_sz;
                const uint8_t current_qv = C_GLOBAL_ORDER.vs_[depth[warp_id]];

#if USE_CUM_PATH_MASK
                smask_t parent_pm = cum_path_mask[warp_id][depth[warp_id] - 1];
#endif
                smask_t* const cur_vb = edge_sm_ptrs[ei * MAX_VCOUNT + current_qv];

                uint32_t read_offset = end_nbr[warp_id][depth[warp_id] - 2];
                if (lane_id == 0) compact_count[warp_id] = 0;
                __syncwarp();

                while (compact_count[warp_id] < WARP_SIZE && read_offset < read_end)
                {
                    uint32_t nbr = UINT32_MAX;
                    if (read_offset + lane_id < read_end)
                    {
                        uint32_t local_off = read_offset + lane_id;
                        if (local_off < base_end)
                            nbr = d_all_local[ei].nbrs_[pre_qe_idx][pre_dv][local_off];
                        else
                            nbr = update_index.nbrs_[pre_qe_idx][pre_dv][local_off - base_end];
#if USE_CUM_PATH_MASK
                        if (cur_vb[nbr] == SMASK_ZERO || (parent_pm & cur_vb[nbr]) == SMASK_ZERO)
#else
                        if (!smask_read(cur_vb, nbr))
#endif
                            nbr = UINT32_MAX;
                        if (nbr == v0 || nbr == v1)
                            nbr = UINT32_MAX;
                        for (uint8_t i = 2u; i < depth[warp_id]; i++)
                            if (result_queue[warp_id][i - 2u][queue_pos[warp_id][i - 2]] == nbr)
                                nbr = UINT32_MAX;
                    }
                    __syncwarp();

                    uint32_t ballot = __ballot_sync(0xffffffff, nbr != UINT32_MAX);
                    uint8_t num_new = __popc(ballot);
                    uint8_t my_rank = __popc(ballot & ((1u << lane_id) - 1));

                    if (nbr != UINT32_MAX && compact_count[warp_id] + my_rank < WARP_SIZE) {
                        compact_nbrs[warp_id][compact_count[warp_id] + my_rank] = nbr;
                        compact_pos[warp_id][compact_count[warp_id] + my_rank] = read_offset + lane_id;
                    }

                    if (lane_id == 0) compact_count[warp_id] += num_new;
                    read_offset += WARP_SIZE;
                    __syncwarp();
                }

                uint32_t num_valid = min(WARP_SIZE, compact_count[warp_id]);
                uint32_t temp_nbr = lane_id < num_valid
                    ? compact_nbrs[warp_id][lane_id]
                    : UINT32_MAX;
                uint32_t temp_pos = lane_id < num_valid
                    ? compact_pos[warp_id][lane_id]
                    : UINT32_MAX;

                if (write_res && *new_res_size >= h_max_new_res_size_)
                    return;

                if (num_valid == 0)
                {
                    if (lane_id == 0)
                    {
                        end_v[warp_id][depth[warp_id] - 2] += 1u;
                        end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                    }
                    __syncwarp();
                } else {
                    if (lane_id == num_valid - 1)
                    {
                        uint32_t total_check_end = min(temp_pos + 1, read_end);
                        if (total_check_end >= read_end) {
                            end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            end_v[warp_id][depth[warp_id] - 2] += 1u;
                        } else {
                            end_nbr[warp_id][depth[warp_id] - 2] = total_check_end;
                        }
                    }
                    __syncwarp();
                }

                bool found = lane_id < num_valid;
                if (lane_id == 0)
                {
                    queue_pos[warp_id][depth[warp_id] - 2] = 0u;
                    queue_size[warp_id][depth[warp_id] - 2] = 0u;
                }
                __syncwarp();
                if (write_res && *new_res_size >= h_max_new_res_size_) {
                    return;
                }
                if (found)
                {
                    // const uint8_t ep0 = C_INDEXING_ORDERS[ei].vs_[0];
                    // const uint8_t ep1 = C_INDEXING_ORDERS[ei].vs_[1];
                    for (uint8_t off = C_GLOBAL_ORDER.bni_offs_[depth[warp_id]] + 1; off < C_GLOBAL_ORDER.bni_offs_[depth[warp_id] + 1]; off++)
                    {
                        const uint8_t& bni = C_GLOBAL_ORDER.bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[bni] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[depth[warp_id]]];
                        const uint32_t& pre_pre_v = bni == 0 ? v0 : (bni == 1 ? v1 : result_queue[warp_id][bni - 2][queue_pos[warp_id][bni - 2]]);
//                         const bool relax_qv = g_d_cf && g_d_cf[ei * C_DV_COUNT + pre_pre_v];
//                         const bool is_ep = (C_GLOBAL_ORDER.vs_[bni] == ep0 || C_GLOBAL_ORDER.vs_[bni] == ep1);
// #if RELAX_DEBUG
//                         // Observational: re-run the binary search the relax would skip, and sample
//                         // the cases where it would have FAILED (unsound skips). Does NOT change the
//                         // skip behavior below, so the over-count is still reproduced.
//                         if (relax_qv && is_ep) {
//                             uint32_t db_base = d_all_local[ei].sizes_[pre_pre_qe_idx][pre_pre_v];
//                             uint32_t db_lb = lower_bound(d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v], db_base, temp_nbr);
//                             bool db_in = (db_lb < db_base) && (d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v][db_lb] == temp_nbr);
//                             if (!db_in && (C_DIR_TO_EDGE[pre_pre_qe_idx] < ei)) {
//                                 uint32_t db_up = update_index.sizes_[pre_pre_qe_idx][pre_pre_v];
//                                 db_lb = lower_bound(update_index.nbrs_[pre_pre_qe_idx][pre_pre_v], db_up, temp_nbr);
//                                 db_in = (db_lb < db_up) && (update_index.nbrs_[pre_pre_qe_idx][pre_pre_v][db_lb] == temp_nbr);
//                             }
//                             if (!db_in && lane_id == 0) {
//                                 uint64_t idx = atomicAdd((unsigned long long*)&g_dbg_skip_fail, 1ULL);
//                                 if (idx < 1) {
//                                     // printf("ei=%u \n", ei);
//                                     printf("[SKIPFAIL] ei=%u depth=%u ep0=%u ep1=%u | v0=%u v1=%u | cur(pos%u)=qv%u temp_nbr=%u | BNpos=%u bn_qv=%u pre_pre_v=%u | gdcf[pre_pre_v]=%u gdcf[temp_nbr]=%u base_sz=%u up_vis=%u | path:",
//                                         (unsigned)ei, (unsigned)depth[warp_id], (unsigned)ep0, (unsigned)ep1,
//                                         (unsigned)v0, (unsigned)v1,
//                                         (unsigned)depth[warp_id], (unsigned)current_qv, (unsigned)temp_nbr,
//                                         (unsigned)bni, (unsigned)C_GLOBAL_ORDER.vs_[bni], (unsigned)pre_pre_v,
//                                         (unsigned)(g_d_cf ? g_d_cf[ei * C_DV_COUNT + pre_pre_v] : 0),
//                                         (unsigned)(g_d_cf ? g_d_cf[ei * C_DV_COUNT + temp_nbr] : 0),
//                                         (unsigned)db_base, (unsigned)(C_DIR_TO_EDGE[pre_pre_qe_idx] < ei));
//                                     for (uint8_t d = 2; d < depth[warp_id]; d++)
//                                         printf(" qv%u=v%u", (unsigned)C_GLOBAL_ORDER.vs_[d],
//                                                (unsigned)result_queue[warp_id][d - 2][queue_pos[warp_id][d - 2]]);
//                                     printf("\n");
//                                 }
//                             }
//                         }
// #endif
//                         if (relax_qv && is_ep) continue;

                        const uint32_t base_sz = d_all_local[ei].sizes_[pre_pre_qe_idx][pre_pre_v];
                        uint32_t lb = lower_bound(d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v], base_sz, temp_nbr);
                        bool in_bn = (lb < base_sz) && (d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v][lb] == temp_nbr);

                        if (!in_bn && (C_DIR_TO_EDGE[pre_pre_qe_idx] < ei)) {
                            const uint32_t up_sz = update_index.sizes_[pre_pre_qe_idx][pre_pre_v];
                            lb = lower_bound(update_index.nbrs_[pre_pre_qe_idx][pre_pre_v], up_sz, temp_nbr);
                            in_bn = (lb < up_sz) && (update_index.nbrs_[pre_pre_qe_idx][pre_pre_v][lb] == temp_nbr);
                        }

                        if (!in_bn)
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
                            if (write_pos + __popc(found_mask) > h_max_new_res_size_) {
                                return;
                            }
                            if (found)
                            {
                                write_pos += rank;
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capability_] = ((uint32_t)ei << 27) | v0;
                                C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capability_] = v1;
                                for (uint8_t j = 2u; j < end_depth - 1; j++)
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capability_] = result_queue[warp_id][j - 2][queue_pos[warp_id][j - 2]];
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
            if (lane_id == 0 && depth[warp_id] < end_depth - 1) {
#if USE_CUM_PATH_MASK
                uint8_t cur_d = depth[warp_id];
                uint32_t vi = result_queue[warp_id][cur_d - 2][queue_pos[warp_id][cur_d - 2]];
                cum_path_mask[warp_id][cur_d] = cum_path_mask[warp_id][cur_d - 1]
                    & edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[cur_d]][vi];
#endif
                depth[warp_id] ++;
            }
            __syncwarp();
        }
    }
}

#ifdef USE_GLOBAL_RQ
__global__ void __launch_bounds__(BLOCK_DIM, 4)
extendBFSAllBitGlobal(
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
) {
    __shared__ uint16_t queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t  queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t  end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ bool     intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t  depth[NWARP_PER_BLOCK];
    __shared__ uint32_t compact_nbrs[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t compact_pos[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t  compact_count[NWARP_PER_BLOCK];
#if USE_CUM_PATH_MASK
    __shared__ smask_t  cum_path_mask[NWARP_PER_BLOCK][MAX_VCOUNT]; // cached cumulative path mask per depth
#endif

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long total_warps = ((unsigned long long)gridDim.x * blockDim.x) / WARP_SIZE;
    const unsigned long long base_gwarp = (unsigned long long)blockIdx.x * blockDim.x / WARP_SIZE + warp_id;

#pragma unroll 1
    for (unsigned long long gwarp_id = base_gwarp;
         gwarp_id < res_size;
         gwarp_id += total_warps)
    {
        const unsigned long long rq_base = (gwarp_id % total_warps) * (C_QV_COUNT - 2) * 256;
        #define RQ(depth_idx, slot) global_rq[rq_base + (depth_idx) * 256 + (slot)]

        uint32_t packed_v0 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capability_];
        uint8_t ei = packed_v0 >> 27;
        uint32_t v0 = packed_v0 & 0x07FFFFFF;
        uint32_t v1 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capability_];

        if (lane_id < start_depth - 2)
            RQ(lane_id, 0) = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 2 + lane_id) % C_RES_QUEUE.capability_];
        __syncwarp();

        if (lane_id < C_QV_COUNT - 2)
        {
            queue_pos[warp_id][lane_id] = 0u;
            queue_size[warp_id][lane_id] = (lane_id < start_depth - 2) ? 1u : 0u;
            end_v[warp_id][lane_id] = 0u;
            end_nbr[warp_id][lane_id] = 0u;
            intersection_continue[warp_id][lane_id] = false;
            compact_count[warp_id] = 0;
        }
        if (lane_id == 0)
        {
            depth[warp_id] = start_depth;
#if USE_CUM_PATH_MASK
            // Initialize cumulative path mask cache
            cum_path_mask[warp_id][0] = edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[0]][v0];
            cum_path_mask[warp_id][1] = cum_path_mask[warp_id][0] & edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[1]][v1];
            for (uint8_t i = 2; i < start_depth; i++) {
                uint32_t vi = RQ(i - 2, 0);
                cum_path_mask[warp_id][i] = cum_path_mask[warp_id][i - 1] & edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[i]][vi];
            }
#endif
        }
        __syncwarp();

        while (depth[warp_id] >= start_depth)
        {
            __syncwarp();
            if (write_res && *new_res_size >= h_max_new_res_size_) return;
            const uint8_t& pre_qv_idx = C_GLOBAL_ORDER.bni_[C_GLOBAL_ORDER.bni_offs_[depth[warp_id]]];
            const uint8_t& pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[pre_qv_idx] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[depth[warp_id]]];
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
                                compact_count[warp_id] = 0;
                            }
                            else
                            {
                                end_v[warp_id][depth[warp_id] - 2] = queue_pos[warp_id][pre_qv_idx - 2];
                                end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                                compact_count[warp_id] = 0;
                            }
                        }
                        __syncwarp();
                    }

                    intersection_continue[warp_id][depth[warp_id] - 2] = true;

                    uint32_t pre_dv = UINT32_MAX;
                    if (pre_qv_idx == 0u)
                        pre_dv = v0;
                    else if (pre_qv_idx == 1u)
                        pre_dv = v1;
                    else
                        pre_dv = RQ(pre_qv_idx - 2, queue_pos[warp_id][pre_qv_idx - 2]);

                    const uint32_t base_end = d_all_local[ei].sizes_[pre_qe_idx][pre_dv];
                    const bool up_vis = (C_DIR_TO_EDGE[pre_qe_idx] < ei);
                    const uint32_t update_sz = up_vis ? update_index.sizes_[pre_qe_idx][pre_dv] : 0;
                    const uint32_t read_end = base_end + update_sz;
                    const uint8_t current_qv = C_GLOBAL_ORDER.vs_[depth[warp_id]];
                    // Per-edge support mask for current query vertex
                    smask_t* const cur_vb = edge_sm_ptrs[ei * MAX_VCOUNT + current_qv];
#if USE_CUM_PATH_MASK
                    // Read cumulative path mask from cache (O(1) shared memory read)
                    smask_t parent_pm = cum_path_mask[warp_id][depth[warp_id] - 1];
#endif
                    uint8_t initial_end_v = end_v[warp_id][depth[warp_id] - 2];

                    uint16_t total_found = 0;
                    // while (total_found + WARP_SIZE <= 256)
                    // {
                        uint32_t read_offset = end_nbr[warp_id][depth[warp_id] - 2];
                        if (lane_id == 0) compact_count[warp_id] = 0;
                        __syncwarp();

                        // Compaction loop
                        while (compact_count[warp_id] < WARP_SIZE && read_offset < read_end)
                        {
                            uint32_t nbr = UINT32_MAX;
                            if (read_offset + lane_id < read_end)
                            {
                                uint32_t local_off = read_offset + lane_id;
                                if (local_off < base_end)
                                    nbr = d_all_local[ei].nbrs_[pre_qe_idx][pre_dv][local_off];
                                else
                                    nbr = update_index.nbrs_[pre_qe_idx][pre_dv][local_off - base_end];
                                // Support mask check + path_mask filtering
#if USE_CUM_PATH_MASK
                                if (cur_vb[nbr] == SMASK_ZERO || (parent_pm & cur_vb[nbr]) == SMASK_ZERO)
#else
                                if (!smask_read(cur_vb, nbr))
#endif
                                    nbr = UINT32_MAX;
                                if (nbr == v0 || nbr == v1)
                                    nbr = UINT32_MAX;
                                for (uint8_t i = 2u; i < depth[warp_id]; i++)
                                    if (RQ(i - 2u, queue_pos[warp_id][i - 2u]) == nbr)
                                        nbr = UINT32_MAX;
                            }
                            __syncwarp();

                            uint32_t ballot_val = __ballot_sync(0xffffffff, nbr != UINT32_MAX);
                            uint8_t num_new = __popc(ballot_val);
                            uint8_t my_rank = __popc(ballot_val & ((1u << lane_id) - 1));

                            if (nbr != UINT32_MAX && compact_count[warp_id] + my_rank < WARP_SIZE) {
                                compact_nbrs[warp_id][compact_count[warp_id] + my_rank] = nbr;
                                compact_pos[warp_id][compact_count[warp_id] + my_rank] = read_offset + lane_id;
                            }

                            if (lane_id == 0) compact_count[warp_id] += num_new;
                            read_offset += WARP_SIZE;
                            __syncwarp();
                        }

                        uint32_t num_valid = min(WARP_SIZE, (uint32_t)compact_count[warp_id]);
                        uint32_t temp_nbr = lane_id < num_valid
                            ? compact_nbrs[warp_id][lane_id]
                            : UINT32_MAX;
                        uint32_t temp_pos = lane_id < num_valid
                            ? compact_pos[warp_id][lane_id]
                            : UINT32_MAX;

                        if (write_res && *new_res_size >= h_max_new_res_size_) return;

                        // Update end_v/end_nbr
                        if (num_valid == 0)
                        {
                            if (lane_id == 0)
                            {
                                end_v[warp_id][depth[warp_id] - 2] += 1u;
                                end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                            }
                            __syncwarp();
                            // break; // parent exhausted
                        }
                        else
                        {
                            if (lane_id == num_valid - 1)
                            {
                                uint32_t total_check_end = min(temp_pos + 1, read_end);
                                if (total_check_end >= read_end) {
                                    end_nbr[warp_id][depth[warp_id] - 2] = 0u;
                                    end_v[warp_id][depth[warp_id] - 2] += 1u;
                                } else {
                                    end_nbr[warp_id][depth[warp_id] - 2] = total_check_end;
                                }
                            }
                            __syncwarp();
                        }

                        // BN check
                        bool found = lane_id < num_valid;
                        if (found)
                        {
                            // const bool relax_qv = g_d_cf && g_d_cf[ei * C_DV_COUNT + temp_nbr] && C_REBUILD_V_FLAGS[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[depth[warp_id]]];
                            // const uint8_t ep0 = C_INDEXING_ORDERS[ei].vs_[0];   // = qe_list_[ei].first  (ei's own endpoint, not global start)
                            // const uint8_t ep1 = C_INDEXING_ORDERS[ei].vs_[1];   // = qe_list_[ei].second
                            for (uint8_t off = C_GLOBAL_ORDER.bni_offs_[depth[warp_id]] + 1; off < C_GLOBAL_ORDER.bni_offs_[depth[warp_id] + 1]; off++)
                            {
                                const uint8_t& bni = C_GLOBAL_ORDER.bni_[off];
                                // if (relax_qv && (C_GLOBAL_ORDER.vs_[bni] == ep0 || C_GLOBAL_ORDER.vs_[bni] == ep1)) continue;  // relax check to ei's own endpoints
                                const uint8_t& pre_pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[bni] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[depth[warp_id]]];
                                const uint32_t pre_pre_v = bni == 0 ? v0 : (bni == 1 ? v1 : RQ(bni - 2, queue_pos[warp_id][bni - 2]));

                                const uint32_t base_sz = d_all_local[ei].sizes_[pre_pre_qe_idx][pre_pre_v];
                                uint32_t lb = lower_bound(d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v], base_sz, temp_nbr);
                                bool in_bn = (lb < base_sz) && (d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v][lb] == temp_nbr);

                                if (!in_bn && (C_DIR_TO_EDGE[pre_pre_qe_idx] < ei)) {
                                    const uint32_t up_sz = update_index.sizes_[pre_pre_qe_idx][pre_pre_v];
                                    lb = lower_bound(update_index.nbrs_[pre_pre_qe_idx][pre_pre_v], up_sz, temp_nbr);
                                    in_bn = (lb < up_sz) && (update_index.nbrs_[pre_pre_qe_idx][pre_pre_v][lb] == temp_nbr);
                                }

                                if (!in_bn)
                                {
                                    found = false;
                                    break;
                                }
                            }
                        }
                        __syncwarp();
                        if (write_res && *new_res_size >= h_max_new_res_size_) return;

                        // Ballot + write results
                        const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                        const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                        const uint8_t found_count = __popc(found_mask);

                        if (depth[warp_id] < end_depth - 1)
                        {
                            if (found) {
                                RQ(depth[warp_id] - 2, total_found + rank) = temp_nbr;
                            }
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
                                        C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capability_] = ((uint32_t)ei << 27) | v0;
                                        C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capability_] = v1;
                                        for (uint8_t j = 2u; j < end_depth - 1; j++)
                                            C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capability_] = RQ(j - 2, queue_pos[warp_id][j - 2]);
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

                        total_found += found_count;

                        // Check if parent exhausted
                        // if (end_v[warp_id][depth[warp_id] - 2] > initial_end_v) break;
                    // }

                    // Set final queue state
                    if (lane_id == 0)
                    {
                        queue_pos[warp_id][depth[warp_id] - 2] = (depth[warp_id] < end_depth - 1) ? 0u : (uint8_t)total_found;
                        queue_size[warp_id][depth[warp_id] - 2] = total_found;
                    }
                    __syncwarp();
                }
            }
            else
            {
                if (lane_id == 0 && depth[warp_id] < end_depth - 1) {
#if USE_CUM_PATH_MASK
                    // Update cumulative path mask cache before going deeper
                    uint8_t cur_d = depth[warp_id];
                    uint32_t vi = RQ(cur_d - 2, queue_pos[warp_id][cur_d - 2]);
                    cum_path_mask[warp_id][cur_d] = cum_path_mask[warp_id][cur_d - 1]
                        & edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[cur_d]][vi];
#endif
                    depth[warp_id] ++;
                }
                __syncwarp();
            }
        }

        #undef RQ
    }
}
#endif
#endif
