#include "index/cpu_index_mirror.h"
#include "graph/graph.h"
#include "graph/plan.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>
#include <vector>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>

// 64-bit edge decoding helpers (encoding: (idx<<59)|(u<<32)|v)
static inline uint32_t edge_idx(uint64_t e) { return (uint32_t)(e >> 59); }
static inline uint32_t edge_u(uint64_t e) { return (uint32_t)((e >> 32) & 0x07FFFFFFUL); }
static inline uint32_t edge_v(uint64_t e) { return (uint32_t)(e & 0xFFFFFFFFUL); }

CPUIndexMirror::CPUIndexMirror(const QueryGraph& query, const Plan& plan, uint32_t num_dir_edges)
    : query_(query), plan_(plan), num_dir_edges_(num_dir_edges)
{
    main_ptr_.resize(num_dir_edges_);
    main_len_.resize(num_dir_edges_);
    main_data_.resize(num_dir_edges_);
    main_live_.resize(num_dir_edges_, 0);
    upd_keys_.resize(num_dir_edges_);
    upd_off_.resize(num_dir_edges_);
    upd_data_.resize(num_dir_edges_);
    BuildDirToEdge();
}

CPUIndexMirror::~CPUIndexMirror() {
    if (sm_host_) cudaFreeHost(sm_host_);
}

void CPUIndexMirror::BuildDirToEdge()
{
    dir_to_edge_.assign(num_dir_edges_, UINT8_MAX);
    for (uint8_t e = 0; e < query_.ecount_; e++) {
        uint32_t d1 = query_.qe_eidx_[e].first;
        uint32_t d2 = query_.qe_eidx_[e].second;
        if (d1 < num_dir_edges_) dir_to_edge_[d1] = e;
        if (d2 < num_dir_edges_) dir_to_edge_[d2] = e;
    }
}

void CPUIndexMirror::EnsureSmBuffer(uint32_t dv_stride)
{
    size_t need = (size_t)MAX_ECOUNT * MAX_VCOUNT * dv_stride;
    if (need <= sm_cap_ && sm_host_) {
        sm_dv_stride_ = dv_stride;  // reuse the reserved buffer; stride/layout updated
        return;
    }
    // Reserve generously (next power of two) like the ReAlloc helper, so the
    // frequent dv_stride growth as vertices activate does NOT trigger a
    // cudaMallocHost/cudaFreeHost (page-locking ~ms) every batch.
    size_t cap = std::max(need, (size_t)8);
    cap = (size_t)std::exp2(std::ceil(std::log2((double)cap)));
    if (sm_host_) cudaFreeHost(sm_host_);
    sm_host_ = nullptr;
    cudaMallocHost((void**)&sm_host_, cap * sizeof(uint32_t));
    sm_cap_ = cap;
    sm_dv_stride_ = dv_stride;
}

void CPUIndexMirror::ResizeForDVCount(uint32_t dv_count)
{
    if (dv_count == dv_count_) return;
    dv_count_ = dv_count;
    // main_ptr_/main_len_ are dense over data vertices: grow, preserving
    // existing entries and zeroing the new tail. Run on the (NUMA-bound) arena
    // so the new pages are first-touched on the CPU DFS node (local reads).
    run([&] {
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, num_dir_edges_, 4),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t idx = r.begin(); idx < r.end(); idx++) {
                    main_ptr_[idx].resize(dv_count, 0u);
                    main_len_[idx].resize(dv_count, 0u);
                }
            });
    });
}

// Group sorted edges by directed-edge idx -> vector of (idx, start, end).
static void group_by_idx(const std::vector<uint64_t>& sorted_edges,
                         std::vector<std::tuple<uint32_t, size_t, size_t>>& groups)
{
    size_t i = 0, n = sorted_edges.size();
    while (i < n) {
        uint32_t idx = edge_idx(sorted_edges[i]);
        size_t s = i;
        while (i < n && edge_idx(sorted_edges[i]) == idx) i++;
        groups.emplace_back(idx, s, i);
    }
}

void CPUIndexMirror::InitFromSortedEdges(const std::vector<uint64_t>& sorted_edges)
{
    run([&] {
        // ptr/len already sized+zeroed by ResizeForDVCount. Clear data per idx.
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, num_dir_edges_, 4),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t idx = r.begin(); idx < r.end(); idx++)
                    main_data_[idx].clear();
            });
        if (sorted_edges.empty()) return;

        std::vector<std::tuple<uint32_t, size_t, size_t>> groups;
        group_by_idx(sorted_edges, groups);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, groups.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t g = r.begin(); g < r.end(); g++) {
                    uint32_t idx = std::get<0>(groups[g]);
                    size_t gs = std::get<1>(groups[g]), ge = std::get<2>(groups[g]);
                    auto& dat = main_data_[idx];
                    auto& ptr = main_ptr_[idx];
                    auto& len = main_len_[idx];
                    for (size_t gi = gs; gi < ge; gi++) {
                        uint32_t u = edge_u(sorted_edges[gi]);
                        if (len[u] == 0)              // first v for this u
                            ptr[u] = (uint32_t)dat.size();
                        dat.push_back(edge_v(sorted_edges[gi]));
                        len[u]++;
                    }
                    main_live_[idx] = dat.size();     // fresh build: no dead space
                }
            });
    });
}

void CPUIndexMirror::MergeIdxMain(uint32_t idx, const uint64_t* edges,
                                  size_t gs, size_t ge, const uint32_t* dv_arr, bool is_insert)
{
    auto& data = main_data_[idx];
    auto& ptr = main_ptr_[idx];
    auto& len = main_len_[idx];

    // In-place append merge: for each touched u, merge its list with the delta
    // and APPEND the result (old list becomes dead space). O(delta + touched),
    // no full O(DV_COUNT) rebuild / realloc. ptr[u]/len[u] updated in place.
    std::vector<uint32_t> tmp;
    int64_t live_delta = 0;
    size_t gi = gs;
    while (gi < ge) {
        uint32_t u = edge_u(edges[gi]);
        size_t ds = gi;
        while (gi < ge && edge_u(edges[gi]) == u) gi++;
        const uint32_t* db = dv_arr + ds;
        size_t dn = gi - ds;
        const uint32_t* ex = data.data() + ptr[u];   // fresh offset each iter
        uint32_t el = len[u];
        tmp.clear();
        if (is_insert)
            std::set_union(ex, ex + el, db, db + dn, std::back_inserter(tmp));
        else
            std::set_difference(ex, ex + el, db, db + dn, std::back_inserter(tmp));
        uint32_t before = (uint32_t)data.size();
        data.insert(data.end(), tmp.begin(), tmp.end());
        live_delta += (int64_t)tmp.size() - (int64_t)el;
        ptr[u] = before;
        len[u] = (uint32_t)tmp.size();
    }
    main_live_[idx] = (live_delta >= 0 || main_live_[idx] > (size_t)(-live_delta))
                      ? main_live_[idx] + live_delta : 0;

    // Compact when dead space exceeds live (append-only data would otherwise grow unbounded).
    if (main_live_[idx] > 0 && data.size() > 2 * main_live_[idx]) {
        std::vector<uint32_t> compacted;
        compacted.reserve(main_live_[idx]);
        for (uint32_t u = 0; u < dv_count_; u++) {
            if (len[u] > 0) {
                uint32_t no = (uint32_t)compacted.size();
                compacted.insert(compacted.end(), data.data() + ptr[u], data.data() + ptr[u] + len[u]);
                ptr[u] = no;
            }
        }
        data.swap(compacted);  // live size unchanged, dead space reclaimed
    }
}

void CPUIndexMirror::MergeInsert(const std::vector<uint64_t>& edges)
{
    if (edges.empty()) return;
    run([&] {
        std::vector<uint64_t> s(edges);
        tbb::parallel_sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());  // matches GPU TransferUpdateAll-ins / BatchUpdateGlobalIndex

        std::vector<uint32_t> dv_arr(s.size());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, s.size(), 4096),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t k = r.begin(); k < r.end(); k++) dv_arr[k] = edge_v(s[k]);
            });
        std::vector<std::tuple<uint32_t, size_t, size_t>> groups;
        group_by_idx(s, groups);
        const uint64_t* ed = s.data();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, groups.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t g = r.begin(); g < r.end(); g++)
                    MergeIdxMain(std::get<0>(groups[g]), ed, std::get<1>(groups[g]),
                                 std::get<2>(groups[g]), dv_arr.data(), true);
            });
    });
}

void CPUIndexMirror::MergeRemove(const std::vector<uint64_t>& edges)
{
    if (edges.empty()) return;
    run([&] {
        std::vector<uint64_t> s(edges);
        tbb::parallel_sort(s.begin(), s.end());  // no unique — matches GPU TransferUpdateAll-del

        std::vector<uint32_t> dv_arr(s.size());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, s.size(), 4096),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t k = r.begin(); k < r.end(); k++) dv_arr[k] = edge_v(s[k]);
            });
        std::vector<std::tuple<uint32_t, size_t, size_t>> groups;
        group_by_idx(s, groups);
        const uint64_t* ed = s.data();
        tbb::parallel_for(tbb::blocked_range<size_t>(0, groups.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t g = r.begin(); g < r.end(); g++)
                    MergeIdxMain(std::get<0>(groups[g]), ed, std::get<1>(groups[g]),
                                 std::get<2>(groups[g]), dv_arr.data(), false);
            });
    });
}

void CPUIndexMirror::RebuildUpdate(const std::vector<uint64_t>& edges)
{
    run([&] {
        // Clear all idx's sparse update structures (O(num_idx) + O(prev delta)).
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, num_dir_edges_, 4),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t idx = r.begin(); idx < r.end(); idx++) {
                    upd_keys_[idx].clear();
                    upd_off_[idx].clear();
                    upd_data_[idx].clear();
                }
            });
        if (edges.empty()) return;

        std::vector<uint64_t> s(edges);
        tbb::parallel_sort(s.begin(), s.end());  // no unique — matches GPU TransferUpdateIndex
        std::vector<std::tuple<uint32_t, size_t, size_t>> groups;
        group_by_idx(s, groups);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, groups.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t g = r.begin(); g < r.end(); g++) {
                    uint32_t idx = std::get<0>(groups[g]);
                    size_t gs = std::get<1>(groups[g]), ge = std::get<2>(groups[g]);
                    auto& keys = upd_keys_[idx];
                    auto& off = upd_off_[idx];
                    auto& dat = upd_data_[idx];
                    for (size_t gi = gs; gi < ge; gi++) {
                        uint32_t u = edge_u(s[gi]);
                        if (off.empty() || keys.back() != u) {  // new u (edges sorted by u)
                            keys.push_back(u);
                            off.push_back((uint32_t)dat.size());
                        }
                        dat.push_back(edge_v(s[gi]));
                    }
                    off.push_back((uint32_t)dat.size());  // sentinel
                }
            });
    });
}
