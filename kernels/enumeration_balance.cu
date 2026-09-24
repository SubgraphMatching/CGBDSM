#include "kernels/enumeration_balance.h"

#ifdef USE_MERGED_MATCHING

#include "utils/search.cuh"
#include "utils/cuda_helpers.h"
#include "utils/globals.h"
#include "graph/graph_gpu.h"

// ── RDMCE-style constants ──
#ifndef TASK_SHARE_BOUND
  #define TASK_SHARE_BOUND 32
#endif

#ifndef STEAL_CHUNK
  #define STEAL_CHUNK 32
#endif

// ═══════════════════════════════════════════════════════════════
// BalanceContext: holds pointers to all shared memory arrays
// ── Initialized once in the kernel, passed to all __device__ functions
// ═══════════════════════════════════════════════════════════════
struct BalanceContext {
    // ── DFS state (2D/3D shared arrays) ──
    uint32_t (*result_queue)[MAX_VCOUNT - 2][WARP_SIZE];
    uint8_t  (*queue_pos)[MAX_VCOUNT - 2];
    uint8_t  (*queue_size)[MAX_VCOUNT - 2];
    uint8_t  (*end_v)[MAX_VCOUNT - 2];
    uint32_t (*end_nbr)[MAX_VCOUNT - 2];
    bool     (*intersection_continue)[MAX_VCOUNT - 2];
    uint8_t  *depth;
    uint32_t (*compact_nbrs)[WARP_SIZE];
    uint32_t (*compact_pos)[WARP_SIZE];
    uint8_t  *compact_count;
    smask_t  (*cum_path_mask)[MAX_VCOUNT];

    // ── WarpMeta: snapshot + coordination ──
    uint32_t *wm_snap_v0;
    uint32_t *wm_snap_v1;
    uint8_t  *wm_snap_ei;
    uint8_t  *wm_snap_depth;
    smask_t  *wm_snap_pm;
    uint32_t (*wm_snap_matching)[MAX_VCOUNT - 2];
    uint8_t  *wm_snap_parent_qe;
    uint32_t *wm_snap_parent_dv;
    uint32_t *wm_snap_read_end;
    uint32_t *wm_snap_base_end;
    int32_t  *wm_task_count;
    int32_t  *wm_pending_count;
    int8_t   *wm_level_sharing;
    int8_t   *wm_level_to_share;
    uint32_t *wm_snap_start_offset; // offset into neighbor array (skip already-scanned)

    // ── Block-level ──
    uint32_t *block_active_count;
};

// ═══════════════════════════════════════════════════════════════
// __device__ PublishExposure: RDMCE-style publish protocol
// ── Write snapshot → pending_count → threadfence → task_count
// ═══════════════════════════════════════════════════════════════
static __device__ __forceinline__
void PublishExposure(
    const BalanceContext& ctx,
    const uint32_t warp_id, const uint8_t lane_id,
    const uint8_t depth_idx,
    const uint32_t v0, const uint32_t v1, const uint8_t ei,
    const smask_t parent_pm,
    const uint32_t pre_dv, const uint8_t pre_qe_idx,
    const uint32_t read_end, const uint32_t base_end,
    const uint32_t start_offset   // skip already-scanned neighbors
) {
    const uint32_t remaining = read_end - start_offset;

    // Step 1a: Scalar snapshot fields — still lane 0 only
    if (lane_id == 0) {
        ctx.wm_snap_v0[warp_id] = v0;
        ctx.wm_snap_v1[warp_id] = v1;
        ctx.wm_snap_ei[warp_id] = ei;
        ctx.wm_snap_depth[warp_id] = depth_idx;
        ctx.wm_snap_pm[warp_id] = parent_pm;
        ctx.wm_snap_parent_dv[warp_id] = pre_dv;
        ctx.wm_snap_parent_qe[warp_id] = pre_qe_idx;
        ctx.wm_snap_read_end[warp_id] = remaining;  // store remaining count
        ctx.wm_snap_base_end[warp_id] = base_end;    // original boundary
        ctx.wm_snap_start_offset[warp_id] = start_offset;
    }

    // Step 1b: Cooperative matching copy — each lane copies one depth level
    if (lane_id < depth_idx)
        ctx.wm_snap_matching[warp_id][lane_id] =
            ctx.result_queue[warp_id][lane_id][ctx.queue_pos[warp_id][lane_id]];
    __syncwarp();

    // Step 2-3: RDMCE release barrier — lane 0 only
    if (lane_id == 0) {
        int32_t num_chunks = (int32_t)((remaining + STEAL_CHUNK - 1) / STEAL_CHUNK);
        ctx.wm_pending_count[warp_id] = num_chunks;
        __threadfence_block();
        ctx.wm_task_count[warp_id] = num_chunks;
        ctx.wm_level_sharing[warp_id] = (int8_t)depth_idx;
    }
}

// ═══════════════════════════════════════════════════════════════
// __device__ SetTaskByChunk: validate neighbors in a stolen chunk
// ── Returns number of valid non-leaf candidates in compact_nbrs
// ═══════════════════════════════════════════════════════════════
static __device__ __forceinline__
uint8_t SetTaskByChunk(
    const BalanceContext& ctx,
    const uint32_t warp_id, const uint8_t lane_id,
    const uint32_t v0, const uint32_t v1, const uint8_t ei,
    const uint8_t snap_depth, const smask_t snap_pm,
    const uint32_t parent_dv, const uint8_t parent_qe,
    const uint32_t chunk_start, const uint32_t chunk_end,
    const uint32_t base_end,
    const uint32_t start_offset,   // add to local_off when reading neighbors
    const uint32_t* __restrict__ snap_matching,
    const RelationsGPU* __restrict__ d_all_local,
    const RelationsGPU update_index,
    smask_t** edge_sm_ptrs,
    const uint8_t end_depth,
    const bool write_res,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_
) {
    // Step 1: Each lane loads one neighbor (offset by start_offset)
    uint32_t nbr = UINT32_MAX;
    if (lane_id < chunk_end - chunk_start) {
        uint32_t local_off = start_offset + chunk_start + lane_id;
        if (local_off < base_end)
            nbr = d_all_local[ei].nbrs_[parent_qe][parent_dv][local_off];
        else
            nbr = update_index.nbrs_[parent_qe][parent_dv][local_off - base_end];
    }

    // Step 2: 3-stage validation
    bool valid = (nbr != UINT32_MAX);
    if (valid) {
        const uint8_t current_qv = C_GLOBAL_ORDER.vs_[snap_depth + 2];
        smask_t* cur_vb = edge_sm_ptrs[ei * MAX_VCOUNT + current_qv];
        if (cur_vb[nbr] == SMASK_ZERO || (snap_pm & cur_vb[nbr]) == SMASK_ZERO)
            valid = false;
    }
    if (valid) {
        if (nbr == v0 || nbr == v1) valid = false;
        else for (uint8_t i = 0; i < snap_depth && valid; i++)
            if (snap_matching[i] == nbr) valid = false;
    }
    if (valid) {
        for (uint8_t off = C_GLOBAL_ORDER.bni_offs_[snap_depth + 2] + 1;
             off < C_GLOBAL_ORDER.bni_offs_[snap_depth + 3] && valid; off++) {
            const uint8_t bni = C_GLOBAL_ORDER.bni_[off];
            const uint8_t bqe = C_EIDX[C_GLOBAL_ORDER.vs_[bni] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[snap_depth + 2]];
            const uint32_t bv = (bni == 0) ? v0 : (bni == 1) ? v1 : snap_matching[bni - 2];
            const uint32_t bsz = d_all_local[ei].sizes_[bqe][bv];
            uint32_t lb = lower_bound(d_all_local[ei].nbrs_[bqe][bv], bsz, nbr);
            bool in_bn = (lb < bsz) && (d_all_local[ei].nbrs_[bqe][bv][lb] == nbr);
            if (!in_bn && (C_DIR_TO_EDGE[bqe] < ei)) {
                const uint32_t usz = update_index.sizes_[bqe][bv];
                lb = lower_bound(update_index.nbrs_[bqe][bv], usz, nbr);
                in_bn = (lb < usz) && (update_index.nbrs_[bqe][bv][lb] == nbr);
            }
            if (!in_bn) valid = false;
        }
    }
    __syncwarp();

    // Step 3: Compact valid neighbors
    const uint32_t valid_mask = __ballot_sync(0xffffffff, valid);
    const uint8_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & valid_mask);
    const uint8_t num_valid = __popc(valid_mask);

    if (snap_depth + 2 >= end_depth - 1) {
        // ── Leaf level ──
        if (write_res && valid_mask) {
            unsigned long long write_pos;
            if (lane_id == 0) write_pos = atomicAdd(new_res_size, (unsigned long long)num_valid);
            write_pos = __shfl_sync(0xffffffff, write_pos, 0);
            if (write_pos + num_valid <= h_max_new_res_size_) {
                if (valid) {
                    write_pos += rank;
                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capability_] = ((uint32_t)ei << 27) | v0;
                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capability_] = v1;
                    for (uint8_t j = 0; j < snap_depth; j++)
                        C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 2 + j) % C_RES_QUEUE.capability_] = snap_matching[j];
                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capability_] = nbr;
                }
            }
        } else if (!write_res) {
            if (valid && rank == 0) atomicAdd(new_res_size, (unsigned long long)num_valid);
        }
        return 0;
    } else {
        if (valid) ctx.compact_nbrs[warp_id][rank] = nbr;
        if (lane_id == 0) ctx.compact_count[warp_id] = num_valid;
        __syncwarp();
        return num_valid;
    }
}

// ═══════════════════════════════════════════════════════════════
// __device__ InitDFSFromCandidate: init DFS state from snapshot + nbr
// ═══════════════════════════════════════════════════════════════
static __device__ __forceinline__
void InitDFSFromCandidate(
    const BalanceContext& ctx,
    const uint32_t warp_id, const uint8_t lane_id,
    const uint32_t v0, const uint32_t v1, const uint8_t ei,
    const uint8_t snap_depth, const uint32_t nbr,
    const smask_t snap_pm,
    const uint32_t* __restrict__ snap_matching,
    smask_t** edge_sm_ptrs,
    const uint8_t start_depth_arg
) {
    if (lane_id < snap_depth)
        ctx.result_queue[warp_id][lane_id][0] = snap_matching[lane_id];
    if (lane_id == snap_depth)
        ctx.result_queue[warp_id][snap_depth][0] = nbr;
    __syncwarp();

    // ── Opt: reuse snap_pm instead of rebuilding from scratch ──
    // The stolen DFS starts at depth = snap_depth + 3, so it only reads
    // cum_path_mask[snap_depth+2] and above. snap_pm already holds the
    // correct cumulative mask at index snap_depth+1.
    if (lane_id == 0) {
        ctx.cum_path_mask[warp_id][snap_depth + 1] = snap_pm;
        const uint8_t current_qv = C_GLOBAL_ORDER.vs_[snap_depth + 2];
        ctx.cum_path_mask[warp_id][snap_depth + 2] = snap_pm
            & edge_sm_ptrs[ei * MAX_VCOUNT + current_qv][nbr];
        ctx.depth[warp_id] = snap_depth + 3;
    }

    if (lane_id < C_QV_COUNT - 2) {
        ctx.queue_pos[warp_id][lane_id] = 0u;
        ctx.queue_size[warp_id][lane_id] = (lane_id <= snap_depth) ? 1u : 0u;
        ctx.end_v[warp_id][lane_id] = 0u;
        ctx.end_nbr[warp_id][lane_id] = 0u;
        ctx.intersection_continue[warp_id][lane_id] = false;
    }
    if (lane_id == 0) ctx.compact_count[warp_id] = 0;
    __syncwarp();
}

// ═══════════════════════════════════════════════════════════════
// __device__ IterKernelBalance: pure DFS loop with publishing
// ── Runs DFS from current state to completion or until publishable depth
// ── Returns published_depth (>=0) if yielded, -1 if completed normally
// ═══════════════════════════════════════════════════════════════
static __device__ __forceinline__
void IterKernelBalance(
    const BalanceContext& ctx,
    const uint32_t warp_id, const uint8_t lane_id,
    const uint32_t v0, const uint32_t v1, const uint8_t ei,
    const uint8_t start_depth_arg, const uint8_t end_depth,
    const RelationsGPU* __restrict__ d_all_local,
    const RelationsGPU update_index,
    smask_t** edge_sm_ptrs,
    const bool write_res,
    const unsigned long long new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size_,
    int8_t& published_depth,
    const bool allow_publish
) {
    published_depth = -1;

    while (ctx.depth[warp_id] >= start_depth_arg)
    {
        __syncwarp();
        if (write_res && *new_res_size >= h_max_new_res_size_) return;

        // Cache depth in register — avoids ~44 shared memory loads per iteration
        const uint8_t cur_depth = ctx.depth[warp_id];
        const uint8_t di = cur_depth - 2;

        const uint8_t& pre_qv_idx = C_GLOBAL_ORDER.bni_[C_GLOBAL_ORDER.bni_offs_[cur_depth]];
        const uint8_t& pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[pre_qv_idx] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[cur_depth]];

        if (ctx.queue_pos[warp_id][di] >= ctx.queue_size[warp_id][di])
        {
            if (ctx.intersection_continue[warp_id][di] &&
                ((pre_qv_idx < 2 && ctx.end_v[warp_id][di] > 0) ||
                 (pre_qv_idx >= 2 && ctx.end_v[warp_id][di] > ctx.queue_pos[warp_id][pre_qv_idx - 2])))
            {
                if (lane_id == 0) {
                    ctx.queue_pos[warp_id][di] = 0u;
                    ctx.queue_size[warp_id][di] = 0u;
                    ctx.intersection_continue[warp_id][di] = false;
                    ctx.depth[warp_id]--;
                    if (ctx.depth[warp_id] >= 2)
                        ctx.queue_pos[warp_id][ctx.depth[warp_id] - 2]++;
                }
                __syncwarp();
            }
            else
            {
                if (!ctx.intersection_continue[warp_id][di]) {
                    if (lane_id == 0) {
                        if (pre_qv_idx < 2) {
                            ctx.end_v[warp_id][di] = 0u;
                            ctx.end_nbr[warp_id][di] = 0u;
                            ctx.compact_count[warp_id] = 0;
                        } else {
                            ctx.end_v[warp_id][di] = ctx.queue_pos[warp_id][pre_qv_idx - 2];
                            ctx.end_nbr[warp_id][di] = 0u;
                            ctx.compact_count[warp_id] = 0;
                        }
                    }
                    __syncwarp();
                }
                ctx.intersection_continue[warp_id][di] = true;

                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u) pre_dv = v0;
                else if (pre_qv_idx == 1u) pre_dv = v1;
                else pre_dv = ctx.result_queue[warp_id][pre_qv_idx - 2][ctx.queue_pos[warp_id][pre_qv_idx - 2]];

                const uint32_t base_end = d_all_local[ei].sizes_[pre_qe_idx][pre_dv];
                const bool up_vis = (C_DIR_TO_EDGE[pre_qe_idx] < ei);
                const uint32_t update_sz = up_vis ? update_index.sizes_[pre_qe_idx][pre_dv] : 0;
                const uint32_t read_end = base_end + update_sz;
                const uint8_t current_qv = C_GLOBAL_ORDER.vs_[cur_depth];

                smask_t parent_pm = ctx.cum_path_mask[warp_id][di + 1];
                smask_t* const cur_vb = edge_sm_ptrs[ei * MAX_VCOUNT + current_qv];

                // ── RDMCE-style deferred publish: record deepest, publish on revisit ──
                // Record the deepest publishable level (don't publish immediately)
                if (allow_publish && read_end >= TASK_SHARE_BOUND && ctx.wm_level_sharing[warp_id] < (int8_t)di) {
                    if (ctx.wm_level_to_share[warp_id] < (int8_t)di)
                        ctx.wm_level_to_share[warp_id] = (int8_t)di;
                }
                // Publish when re-visiting the deepest publishable level after backtrack
                // (end_nbr > 0 means we've already scanned at least one batch here)
                if (allow_publish && ctx.wm_level_to_share[warp_id] == (int8_t)di
                    && ctx.wm_level_sharing[warp_id] < (int8_t)di
                    && ctx.end_nbr[warp_id][di] > 0
                    && read_end - ctx.end_nbr[warp_id][di] >= TASK_SHARE_BOUND) {
                    PublishExposure(ctx, warp_id, lane_id,
                        di, v0, v1, ei,
                        parent_pm, pre_dv, pre_qe_idx,
                        read_end, base_end, ctx.end_nbr[warp_id][di]);
                    published_depth = (int8_t)di;
                    ctx.wm_level_to_share[warp_id] = (int8_t)di;
                    return;
                }

                // ── Normal neighbor scanning ──
                uint32_t read_offset = ctx.end_nbr[warp_id][di];
                if (lane_id == 0) ctx.compact_count[warp_id] = 0;
                __syncwarp();

                uint8_t cc = 0;  // register-cached compact_count
                while (cc < WARP_SIZE && read_offset < read_end)
                {
                    uint32_t nbr = UINT32_MAX;
                    if (read_offset + lane_id < read_end) {
                        uint32_t local_off = read_offset + lane_id;
                        if (local_off < base_end)
                            nbr = d_all_local[ei].nbrs_[pre_qe_idx][pre_dv][local_off];
                        else
                            nbr = update_index.nbrs_[pre_qe_idx][pre_dv][local_off - base_end];
                        if (cur_vb[nbr] == SMASK_ZERO || (parent_pm & cur_vb[nbr]) == SMASK_ZERO)
                            nbr = UINT32_MAX;
                        if (nbr == v0 || nbr == v1) nbr = UINT32_MAX;
                        for (uint8_t i = 2u; i < cur_depth; i++)
                            if (ctx.result_queue[warp_id][i - 2u][ctx.queue_pos[warp_id][i - 2]] == nbr)
                                nbr = UINT32_MAX;
                    }
                    __syncwarp();

                    uint32_t ballot = __ballot_sync(0xffffffff, nbr != UINT32_MAX);
                    uint8_t num_new = __popc(ballot);
                    uint8_t my_rank = __popc(ballot & ((1u << lane_id) - 1));

                    if (nbr != UINT32_MAX && cc + my_rank < WARP_SIZE) {
                        ctx.compact_nbrs[warp_id][cc + my_rank] = nbr;
                        ctx.compact_pos[warp_id][cc + my_rank] = read_offset + lane_id;
                    }
                    cc += num_new;
                    if (lane_id == 0) ctx.compact_count[warp_id] = cc;
                    read_offset += WARP_SIZE;
                    __syncwarp();
                }

                uint32_t num_valid = min(WARP_SIZE, (uint32_t)cc);
                uint32_t temp_nbr = lane_id < num_valid ? ctx.compact_nbrs[warp_id][lane_id] : UINT32_MAX;
                uint32_t temp_pos = lane_id < num_valid ? ctx.compact_pos[warp_id][lane_id] : UINT32_MAX;

                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                // ── Opt 3: merged syncwarp — one barrier instead of three ──
                if (num_valid == 0) {
                    if (lane_id == 0) {
                        ctx.end_v[warp_id][di] += 1u;
                        ctx.end_nbr[warp_id][di] = 0u;
                    }
                } else {
                    if (lane_id == num_valid - 1) {
                        uint32_t total_check_end = min(temp_pos + 1, read_end);
                        if (total_check_end >= read_end) {
                            ctx.end_nbr[warp_id][di] = 0u;
                            ctx.end_v[warp_id][di] += 1u;
                        } else {
                            ctx.end_nbr[warp_id][di] = total_check_end;
                        }
                    }
                }
                bool found = lane_id < num_valid;
                if (lane_id == 0) {
                    ctx.queue_pos[warp_id][di] = 0u;
                    ctx.queue_size[warp_id][di] = 0u;
                }
                __syncwarp();

                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                if (found) {
                    for (uint8_t off = C_GLOBAL_ORDER.bni_offs_[cur_depth] + 1;
                         off < C_GLOBAL_ORDER.bni_offs_[cur_depth + 1]; off++) {
                        const uint8_t& bni = C_GLOBAL_ORDER.bni_[off];
                        const uint8_t& pre_pre_qe_idx = C_EIDX[C_GLOBAL_ORDER.vs_[bni] * C_QV_COUNT + C_GLOBAL_ORDER.vs_[cur_depth]];
                        const uint32_t& pre_pre_v = bni == 0 ? v0 : (bni == 1 ? v1 : ctx.result_queue[warp_id][bni - 2][ctx.queue_pos[warp_id][bni - 2]]);

                        const uint32_t base_sz = d_all_local[ei].sizes_[pre_pre_qe_idx][pre_pre_v];
                        uint32_t lb = lower_bound(d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v], base_sz, temp_nbr);
                        bool in_bn = (lb < base_sz) && (d_all_local[ei].nbrs_[pre_pre_qe_idx][pre_pre_v][lb] == temp_nbr);
                        if (!in_bn && (C_DIR_TO_EDGE[pre_pre_qe_idx] < ei)) {
                            const uint32_t up_sz = update_index.sizes_[pre_pre_qe_idx][pre_pre_v];
                            lb = lower_bound(update_index.nbrs_[pre_pre_qe_idx][pre_pre_v], up_sz, temp_nbr);
                            in_bn = (lb < up_sz) && (update_index.nbrs_[pre_pre_qe_idx][pre_pre_v][lb] == temp_nbr);
                        }
                        if (!in_bn) { found = false; break; }
                    }
                }
                if (write_res && *new_res_size >= h_max_new_res_size_) return;

                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);

                if (cur_depth < end_depth - 1) {
                    if (found) ctx.result_queue[warp_id][di][rank] = temp_nbr;
                } else {
                    if (write_res) {
                        if (found_mask) {
                            unsigned long long int write_pos;
                            if (lane_id == 0) write_pos = atomicAdd(new_res_size, __popc(found_mask));
                            write_pos = __shfl_sync(0xffffffff, write_pos, 0);
                            if (write_pos + __popc(found_mask) <= h_max_new_res_size_) {
                                if (found) {
                                    write_pos += rank;
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth) % C_RES_QUEUE.capability_] = ((uint32_t)ei << 27) | v0;
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + 1) % C_RES_QUEUE.capability_] = v1;
                                    for (uint8_t j = 2u; j < end_depth - 1; j++)
                                        C_RES_QUEUE.array_[(new_res + write_pos * end_depth + j) % C_RES_QUEUE.capability_] = ctx.result_queue[warp_id][j - 2][ctx.queue_pos[warp_id][j - 2]];
                                    C_RES_QUEUE.array_[(new_res + write_pos * end_depth + end_depth - 1) % C_RES_QUEUE.capability_] = temp_nbr;
                                }
                            }
                        }
                    } else {
                        if (found && rank == 0) atomicAdd(new_res_size, __popc(found_mask));
                    }
                }
                if (found && rank == 0) {
                    ctx.queue_pos[warp_id][di] = (cur_depth < end_depth - 1) ? 0u : __popc(found_mask);
                    ctx.queue_size[warp_id][di] = __popc(found_mask);
                }
                __syncwarp();
            }
        }
        else
        {
            if (lane_id == 0 && cur_depth < end_depth - 1) {
                uint32_t vi = ctx.result_queue[warp_id][di][ctx.queue_pos[warp_id][di]];
                ctx.cum_path_mask[warp_id][cur_depth] = ctx.cum_path_mask[warp_id][di + 1]
                    & edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[cur_depth]][vi];
                ctx.depth[warp_id]++;
            }
            __syncwarp();
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// __global__ extendBFSAllBitBalance
// ═══════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(BLOCK_DIM, 2)
extendBFSAllBitBalance(
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
    // ── DFS shared memory ──
    __shared__ uint32_t result_queue[NWARP_PER_BLOCK][MAX_VCOUNT - 2][WARP_SIZE];
    __shared__ uint8_t  queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t  queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t  end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ bool     intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t  depth[NWARP_PER_BLOCK];
    __shared__ uint32_t compact_nbrs[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t compact_pos[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint8_t  compact_count[NWARP_PER_BLOCK];
    __shared__ smask_t  cum_path_mask[NWARP_PER_BLOCK][MAX_VCOUNT];

    // ── WarpMeta ──
    __shared__ uint32_t wm_snap_v0[NWARP_PER_BLOCK];
    __shared__ uint32_t wm_snap_v1[NWARP_PER_BLOCK];
    __shared__ uint8_t  wm_snap_ei[NWARP_PER_BLOCK];
    __shared__ uint8_t  wm_snap_depth[NWARP_PER_BLOCK];
    __shared__ smask_t  wm_snap_pm[NWARP_PER_BLOCK];
    __shared__ uint32_t wm_snap_matching[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t  wm_snap_parent_qe[NWARP_PER_BLOCK];
    __shared__ uint32_t wm_snap_parent_dv[NWARP_PER_BLOCK];
    __shared__ uint32_t wm_snap_read_end[NWARP_PER_BLOCK];
    __shared__ uint32_t wm_snap_base_end[NWARP_PER_BLOCK];
    __shared__ int32_t  wm_task_count[NWARP_PER_BLOCK];
    __shared__ int32_t  wm_pending_count[NWARP_PER_BLOCK];
    __shared__ int8_t   wm_level_sharing[NWARP_PER_BLOCK];
    __shared__ int8_t   wm_level_to_share[NWARP_PER_BLOCK];
    __shared__ uint32_t wm_snap_start_offset[NWARP_PER_BLOCK];
    __shared__ uint32_t block_active_count;

    // ── Thief neighbor buffer (avoids compact_nbrs corruption by IterKernelBalance) ──
    __shared__ uint32_t thief_nbrs[NWARP_PER_BLOCK][WARP_SIZE];

    // ── Build context struct ──
    BalanceContext ctx;
    ctx.result_queue          = result_queue;
    ctx.queue_pos             = queue_pos;
    ctx.queue_size            = queue_size;
    ctx.end_v                 = end_v;
    ctx.end_nbr               = end_nbr;
    ctx.intersection_continue = intersection_continue;
    ctx.depth                 = depth;
    ctx.compact_nbrs          = compact_nbrs;
    ctx.compact_pos           = compact_pos;
    ctx.compact_count         = compact_count;
    ctx.cum_path_mask         = cum_path_mask;
    ctx.wm_snap_v0            = wm_snap_v0;
    ctx.wm_snap_v1            = wm_snap_v1;
    ctx.wm_snap_ei            = wm_snap_ei;
    ctx.wm_snap_depth         = wm_snap_depth;
    ctx.wm_snap_pm            = wm_snap_pm;
    ctx.wm_snap_matching      = wm_snap_matching;
    ctx.wm_snap_parent_qe     = wm_snap_parent_qe;
    ctx.wm_snap_parent_dv     = wm_snap_parent_dv;
    ctx.wm_snap_read_end      = wm_snap_read_end;
    ctx.wm_snap_base_end      = wm_snap_base_end;
    ctx.wm_task_count         = wm_task_count;
    ctx.wm_pending_count      = wm_pending_count;
    ctx.wm_level_sharing      = wm_level_sharing;
    ctx.wm_level_to_share     = wm_level_to_share;
    ctx.wm_snap_start_offset  = wm_snap_start_offset;
    ctx.block_active_count    = &block_active_count;

    // ── Thread identification ──
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint8_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long) blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    const bool is_idle = (gwarp_id >= res_size);

    // ── Initialize WarpMeta ──
    if (lane_id == 0) {
        wm_task_count[warp_id] = 0;
        wm_pending_count[warp_id] = 0;
        wm_level_sharing[warp_id] = -1;
        wm_level_to_share[warp_id] = -1;
    }
    if (warp_id == 0 && lane_id == 0) {
        unsigned long long total_in_block = res_size - (unsigned long long)blockIdx.x * NWARP_PER_BLOCK;
        block_active_count = (uint32_t)min((unsigned long long)NWARP_PER_BLOCK, total_in_block);
    }
    __syncthreads();

    // ── Load initial matching ──
    uint32_t v0 = 0, v1 = 0;
    uint8_t ei = 0;
    if (!is_idle) {
        uint32_t packed_v0 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth) % C_RES_QUEUE.capability_];
        ei = packed_v0 >> 27;
        v0 = packed_v0 & 0x07FFFFFF;
        v1 = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 1) % C_RES_QUEUE.capability_];
        if (lane_id < start_depth - 2)
            result_queue[warp_id][lane_id][0] = C_RES_QUEUE.array_[(res + gwarp_id * start_depth + 2 + lane_id) % C_RES_QUEUE.capability_];
    }
    __syncwarp();

    // ── Initialize DFS state ──
    if (lane_id < C_QV_COUNT - 2) {
        queue_pos[warp_id][lane_id] = 0u;
        queue_size[warp_id][lane_id] = (!is_idle && lane_id < start_depth - 2) ? 1u : 0u;
        end_v[warp_id][lane_id] = 0u;
        end_nbr[warp_id][lane_id] = 0u;
        intersection_continue[warp_id][lane_id] = false;
        compact_count[warp_id] = 0;
    }
    if (lane_id == 0) {
        depth[warp_id] = start_depth;
    }
    // ── Warp-cooperative cum_path_mask init via prefix-AND scan ──
    if (!is_idle) {
        smask_t my_mask = SMASK_ALL;
        if (lane_id == 0)
            my_mask = edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[0]][v0];
        else if (lane_id == 1)
            my_mask = edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[1]][v1];
        else if (lane_id < start_depth)
            my_mask = edge_sm_ptrs[ei * MAX_VCOUNT + C_GLOBAL_ORDER.vs_[lane_id]][result_queue[warp_id][lane_id - 2][0]];
        __syncwarp();

        // Inclusive prefix-AND scan across first start_depth lanes
        for (int d = 1; d < start_depth; d *= 2) {
            smask_t n = __shfl_up_sync(0xffffffff, my_mask, d);
            if (lane_id >= d) my_mask &= n;
        }
        if (lane_id < start_depth)
            cum_path_mask[warp_id][lane_id] = my_mask;
    }
    __syncwarp();

    // ════════════════════════════════════════════════
    // Phase A: Owner DFS
    // ════════════════════════════════════════════════
    if (!is_idle) {
        int8_t published_depth = -1;
        IterKernelBalance(ctx, warp_id, lane_id, v0, v1, ei,
            start_depth, end_depth, d_all_local, update_index, edge_sm_ptrs,
            write_res, new_res, new_res_size, h_max_new_res_size_,
            published_depth, true);

        // ── Self-consumption ──
        if (published_depth >= 0) {
            while (true) {
                int32_t tc;
                if (lane_id == 0) tc = wm_task_count[warp_id];
                tc = __shfl_sync(0xffffffff, tc, 0);
                if (tc <= 0) {
                    int32_t pc;
                    if (lane_id == 0) pc = *((volatile int32_t*)&wm_pending_count[warp_id]);
                    pc = __shfl_sync(0xffffffff, pc, 0);
                    if (pc <= 0) break;
                    __nanosleep(50);
                    continue;
                }
                bool claimed = false;
                int32_t chunk_id = 0;
                if (lane_id == 0) {
                    int32_t old = atomicCAS(&wm_task_count[warp_id], tc, tc - 1);
                    if (old == tc) { claimed = true; chunk_id = tc - 1; }
                }
                claimed = __shfl_sync(0xffffffff, claimed, 0);
                if (!claimed) continue;
                chunk_id = __shfl_sync(0xffffffff, chunk_id, 0);

                uint32_t cs = (uint32_t)chunk_id * STEAL_CHUNK;
                uint32_t ce = min(cs + (uint32_t)STEAL_CHUNK, wm_snap_read_end[warp_id]);
                ce = __shfl_sync(0xffffffff, ce, 0);

                uint8_t num_valid = SetTaskByChunk(ctx, warp_id, lane_id,
                    wm_snap_v0[warp_id], wm_snap_v1[warp_id], wm_snap_ei[warp_id],
                    wm_snap_depth[warp_id], wm_snap_pm[warp_id],
                    wm_snap_parent_dv[warp_id], wm_snap_parent_qe[warp_id],
                    cs, ce, wm_snap_base_end[warp_id],
                    wm_snap_start_offset[warp_id],
                    wm_snap_matching[warp_id],
                    d_all_local, update_index, edge_sm_ptrs,
                    end_depth, write_res, new_res, new_res_size, h_max_new_res_size_);

                // ── Save valid neighbors to thief_nbrs before DFS overwrites compact_nbrs ──
                if (lane_id == 0)
                    for (uint8_t j = 0; j < num_valid; j++)
                        thief_nbrs[warp_id][j] = compact_nbrs[warp_id][j];
                __syncwarp();
                uint8_t saved_count = num_valid;

                for (uint8_t vi = 0; vi < saved_count; vi++) {
                    uint32_t nbr;
                    if (lane_id == 0) nbr = thief_nbrs[warp_id][vi];
                    nbr = __shfl_sync(0xffffffff, nbr, 0);
                    InitDFSFromCandidate(ctx, warp_id, lane_id,
                        wm_snap_v0[warp_id], wm_snap_v1[warp_id], wm_snap_ei[warp_id],
                        wm_snap_depth[warp_id], nbr, wm_snap_pm[warp_id],
                        wm_snap_matching[warp_id],
                        edge_sm_ptrs, start_depth);

                    int8_t dummy = -1;
                    IterKernelBalance(ctx, warp_id, lane_id,
                        wm_snap_v0[warp_id], wm_snap_v1[warp_id], wm_snap_ei[warp_id],
                        wm_snap_depth[warp_id] + 3, end_depth, d_all_local, update_index, edge_sm_ptrs,
                        write_res, new_res, new_res_size, h_max_new_res_size_, dummy, false);
                }
                __threadfence_block();
                if (lane_id == 0) atomicSub(&wm_pending_count[warp_id], 1);
            }
        }

        // ── Wait for thieves ──
        if (lane_id == 0 && wm_level_sharing[warp_id] >= 0) {
            while (*((volatile int32_t*)&wm_pending_count[warp_id]) > 0)
                __nanosleep(50);
        }
        __syncwarp();
        if (lane_id == 0) atomicSub(&block_active_count, 1u);
        __threadfence_block();
    }

    // ════════════════════════════════════════════════
    // Phase B: Cross-warp stealing
    // ════════════════════════════════════════════════
    while (true) {
        bool exit_flag;
        if (lane_id == 0) {
            __threadfence_block();
            exit_flag = (*((volatile uint32_t*)&block_active_count) == 0);
        }
        exit_flag = __shfl_sync(0xffffffff, exit_flag, 0);
        if (exit_flag) break;

        uint32_t target = (warp_id + 1) % NWARP_PER_BLOCK;
        bool found = false;

        #pragma unroll 1
        for (uint32_t attempt = 0; attempt < NWARP_PER_BLOCK; attempt++) {
            if (found) break;

            int32_t tc;
            if (lane_id == 0) tc = *((volatile int32_t*)&wm_task_count[target]);
            tc = __shfl_sync(0xffffffff, tc, 0);
            if (tc <= 0) { target = (target + 1) % NWARP_PER_BLOCK; continue; }

            bool claimed = false; int32_t chunk_id = 0;
            if (lane_id == 0) {
                int32_t old = atomicCAS(&wm_task_count[target], tc, tc - 1);
                if (old == tc) { claimed = true; chunk_id = tc - 1; }
            }
            claimed = __shfl_sync(0xffffffff, claimed, 0);
            if (!claimed) { target = (target + 1) % NWARP_PER_BLOCK; continue; }
            chunk_id = __shfl_sync(0xffffffff, chunk_id, 0);

            // Copy victim snapshot to own wm_snap_*[warp_id] — warp-cooperative
            if (lane_id == 0) wm_snap_v0[warp_id] = wm_snap_v0[target];
            if (lane_id == 1) wm_snap_v1[warp_id] = wm_snap_v1[target];
            if (lane_id == 2) wm_snap_ei[warp_id] = wm_snap_ei[target];
            if (lane_id == 3) wm_snap_depth[warp_id] = wm_snap_depth[target];
            if (lane_id == 4) wm_snap_pm[warp_id] = wm_snap_pm[target];
            if (lane_id == 5) wm_snap_parent_dv[warp_id] = wm_snap_parent_dv[target];
            if (lane_id == 6) wm_snap_parent_qe[warp_id] = wm_snap_parent_qe[target];
            if (lane_id == 7) wm_snap_read_end[warp_id] = wm_snap_read_end[target];
            if (lane_id == 8) wm_snap_base_end[warp_id] = wm_snap_base_end[target];
            if (lane_id == 9) wm_snap_start_offset[warp_id] = wm_snap_start_offset[target];
            if (lane_id < wm_snap_depth[target])
                wm_snap_matching[warp_id][lane_id] = wm_snap_matching[target][lane_id];
            __syncwarp();

            uint32_t cs = (uint32_t)chunk_id * STEAL_CHUNK;
            uint32_t ce = min(cs + (uint32_t)STEAL_CHUNK, wm_snap_read_end[warp_id]);

            uint8_t num_valid = SetTaskByChunk(ctx, warp_id, lane_id,
                wm_snap_v0[warp_id], wm_snap_v1[warp_id], wm_snap_ei[warp_id],
                wm_snap_depth[warp_id], wm_snap_pm[warp_id],
                wm_snap_parent_dv[warp_id], wm_snap_parent_qe[warp_id],
                cs, ce, wm_snap_base_end[warp_id],
                wm_snap_start_offset[warp_id],
                wm_snap_matching[warp_id],
                d_all_local, update_index, edge_sm_ptrs,
                end_depth, write_res, new_res, new_res_size, h_max_new_res_size_);

            // ── Save valid neighbors to thief_nbrs before DFS overwrites compact_nbrs ──
            if (lane_id == 0)
                for (uint8_t j = 0; j < num_valid; j++)
                    thief_nbrs[warp_id][j] = compact_nbrs[warp_id][j];
            uint8_t saved_count = num_valid;
            if (lane_id == 0) atomicAdd(&block_active_count, 1u);
            __threadfence_block(); __syncwarp();

            for (uint8_t vi = 0; vi < saved_count; vi++) {
                uint32_t nbr;
                if (lane_id == 0) nbr = thief_nbrs[warp_id][vi];
                nbr = __shfl_sync(0xffffffff, nbr, 0);

                InitDFSFromCandidate(ctx, warp_id, lane_id,
                    wm_snap_v0[warp_id], wm_snap_v1[warp_id], wm_snap_ei[warp_id],
                    wm_snap_depth[warp_id], nbr, wm_snap_pm[warp_id],
                    wm_snap_matching[warp_id],
                    edge_sm_ptrs, start_depth);

                int8_t dummy = -1;
                IterKernelBalance(ctx, warp_id, lane_id,
                    wm_snap_v0[warp_id], wm_snap_v1[warp_id], wm_snap_ei[warp_id],
                    wm_snap_depth[warp_id] + 3, end_depth,
                    d_all_local, update_index, edge_sm_ptrs,
                    write_res, new_res, new_res_size, h_max_new_res_size_, dummy, false);
            }
            if (lane_id == 0) {
                atomicSub(&block_active_count, 1u);
                atomicSub(&wm_pending_count[target], 1);
            }
            __threadfence_block(); __syncwarp();
            found = true;
        }
        if (!found) __nanosleep(50);
    }
}

#endif // USE_MERGED_MATCHING
