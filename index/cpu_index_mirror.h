#ifndef INDEX_CPU_INDEX_MIRROR_H
#define INDEX_CPU_INDEX_MIRROR_H

#include <cstdint>
#include <vector>
#include <algorithm>
#include <tbb/task_arena.h>
#include "utils/config.h"

class QueryGraph;
class Plan;

// ============================================================================
// CPUIndexMirror: a host-side mirror of the GPU index's sorted neighbor lists,
// kept in sync via sorted merge from the SAME host staging edge-lists the GPU
// update pipeline consumes (el_host_stage_ / del_host_stage_ / G_UPDATE_FLAT_ /
// el_update_host_stage_).
//
// Storage (O(delta) merge, fast reads):
//  - main_ (incremental): dense ptr+len per data vertex (O(1) read) PLUS a
//    sorted list of non-empty u ("keys"). Merge only iterates keys ∪ delta_u,
//    so it is O(non-empty + delta) work + a cheap dense len memset — never a
//    full O(DV_COUNT) scan of empty vertices.
//  - update_ (rebuilt every batch): fully sparse (keys + offsets + flat data),
//    so RebuildUpdate is O(delta); UpdateList does a binary search.
//
// The CPU DFS does NOT need flat_support_masks_ (pure prune). Acceptance is
// backward-edge binary search + injectivity, so only neighbor lists are stored.
//
// 64-bit encoding: (gamma_eidx << 59) | (new_u << 32) | new_v, sorted ascending.
// ============================================================================
class CPUIndexMirror
{
public:
    CPUIndexMirror(const QueryGraph& query, const Plan& plan, uint32_t num_dir_edges);
    ~CPUIndexMirror();

    // Run all internal parallel work (sort + parallel_for) on this NUMA-bound
    // arena (the cpu_dfs_arena_) instead of the default implicit arena. Must be
    // set before any merge if you want NUMA-local, dedicated workers.
    void SetArena(tbb::task_arena* a) { arena_ = a; }

    void ResizeForDVCount(uint32_t dv_count);

    void InitFromSortedEdges(const std::vector<uint64_t>& sorted_edges);
    void MergeInsert(const std::vector<uint64_t>& edges);
    void MergeRemove(const std::vector<uint64_t>& edges);
    void RebuildUpdate(const std::vector<uint64_t>& edges);

    // ---- DFS accessors (signatures unchanged from the dense-offset version) ----
    // main_: O(1) dense ptr+len.
    inline const uint32_t* MainList(uint32_t idx, uint32_t u, uint32_t& len) const
    {
        len = main_len_[idx][u];
        return main_data_[idx].data() + main_ptr_[idx][u];
    }
    // update_: O(log) binary search over the sparse key list.
    inline const uint32_t* UpdateList(uint32_t idx, uint32_t u, uint32_t& len) const
    {
        len = 0;
        const auto& keys = upd_keys_[idx];
        auto it = std::lower_bound(keys.begin(), keys.end(), u);
        if (it != keys.end() && *it == u) {
            size_t p = (size_t)(it - keys.begin());
            uint32_t s = upd_off_[idx][p];
            len = upd_off_[idx][p + 1] - s;
            return upd_data_[idx].data() + s;
        }
        return nullptr;
    }

    inline uint32_t dv_count() const { return dv_count_; }
    inline uint32_t num_dir_edges() const { return num_dir_edges_; }
    inline uint8_t DirToEdge(uint32_t idx) const { return dir_to_edge_[idx]; }

    // ---- support-mask (the GPU's flat_support_masks_) ----
    // The CPU DFS uses this as a cheap superset prune BEFORE the backward-edge
    // binary searches (mirrors the GPU's smask_read). It is count-preserving
    // (false positives allowed, never false negatives for rebuilt vertices).
    // Layout matches the GPU: tile (ei,qv) at offset (ei*MAX_VCOUNT+qv)*stride.
    void EnsureSmBuffer(uint32_t dv_stride);
    inline uint32_t* SmHostPtr() { return sm_host_; }
    inline uint32_t SmDvStride() const { return sm_dv_stride_; }
    // Pointer to the mask word-array for (ei, qv); nullptr if no mask synced.
    inline const uint32_t* SmPtr(uint32_t ei, uint32_t qv) const {
        return sm_host_ ? sm_host_ + (size_t)(ei * MAX_VCOUNT + qv) * sm_dv_stride_ : nullptr;
    }

private:
    const QueryGraph& query_;
    const Plan& plan_;
    uint32_t num_dir_edges_;
    uint32_t dv_count_ = 0;
    tbb::task_arena* arena_ = nullptr;  // optional NUMA-bound arena for parallel work

    // main_ (incremental): dense ptr/len over data vertices + flat data.
    std::vector<std::vector<uint32_t>> main_ptr_;   // [idx][u] -> offset in main_data_[idx]
    std::vector<std::vector<uint32_t>> main_len_;   // [idx][u] -> length (0 = empty)
    std::vector<std::vector<uint32_t>> main_data_;  // [idx] flat list data (append-only + periodic compact)
    std::vector<size_t> main_live_;                 // [idx] live (non-dead) data size, for compaction

    // update_ (rebuilt per batch): sparse.
    std::vector<std::vector<uint32_t>> upd_keys_;   // [idx] sorted u with data
    std::vector<std::vector<uint32_t>> upd_off_;    // [idx] offsets (size keys+1)
    std::vector<std::vector<uint32_t>> upd_data_;   // [idx] flat list data

    std::vector<uint8_t> dir_to_edge_;

    // Support-mask host mirror (pinned): MAX_ECOUNT*MAX_VCOUNT tiles of sm_dv_stride_ words.
    uint32_t* sm_host_ = nullptr;
    size_t sm_cap_ = 0;          // in uint32 words
    uint32_t sm_dv_stride_ = 0;

    void BuildDirToEdge();

    // Merge a sorted 64-bit delta into main_ for one touched idx (is_insert:
    // set_union, else set_difference). Reads/writes main_*[idx].
    void MergeIdxMain(uint32_t idx, const uint64_t* edges, size_t g_start, size_t g_end,
                      const uint32_t* dv_arr, bool is_insert);

    // Run a lambda on the NUMA-bound arena if set, else on the current thread.
    template <class F>
    void run(F&& f) { if (arena_) arena_->execute(f); else f(); }
};

#endif // INDEX_CPU_INDEX_MIRROR_H
