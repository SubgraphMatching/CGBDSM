#include "index/cpu_dfs.h"
#include "graph/graph.h"
#include "graph/plan.h"
#include "utils/config.h"
#include "utils/constants.h"

#include <algorithm>
#include <cstring>
#ifdef ENABLE_CPU_DFS
#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

namespace {

// Host equivalent of utils/search.cuh lower_bound: returns index of first
// element >= v, or `size` if not present / all smaller.
inline uint32_t lb_index(const uint32_t* a, uint32_t size, uint32_t v)
{
    if (size == 0) return size;
    const uint32_t* it = std::lower_bound(a, a + size, v);
    return (uint32_t)(it - a);
}

// Does sorted list (base[0..bl) then app[0..al)) contain v? Matches the GPU's
// "search d_all_local then update_index" two-step binary search.
inline bool list_contains(const uint32_t* base, uint32_t bl,
                          const uint32_t* app, uint32_t al, uint32_t v)
{
    if (bl) {
        uint32_t p = lb_index(base, bl, v);
        if (p < bl && base[p] == v) return true;
    }
    if (al) {
        uint32_t p = lb_index(app, al, v);
        if (p < al && app[p] == v) return true;
    }
    return false;
}

struct DFSRunner {
    const CPUIndexMirror& mirror;
    const QueryGraph& query;
    const Plan& plan;
    const OrderPerEdge& go;   // plan.global_order_
    uint8_t ei;
    uint32_t seed_dir_a;      // eidx[io.vs_[0], io.vs_[1]] for ei's indexing order
    uint32_t seed_dir_b;      // eidx[io.vs_[1], io.vs_[0]]
    uint32_t matched[MAX_VCOUNT];

    DFSRunner(const CPUIndexMirror& m, const QueryGraph& q, const Plan& p)
        : mirror(m), query(q), plan(p), go(p.global_order_), ei(0),
          seed_dir_a(UINT32_MAX), seed_dir_b(UINT32_MAX)
    {
        std::memset(matched, 0, sizeof(matched));
    }

    inline bool is_seed(uint32_t qe_idx) const
    {
        return qe_idx == seed_dir_a || qe_idx == seed_dir_b;
    }

    // Resolve (base, app) for a directed edge qe_idx at data vertex dv under ei.
    // base = update list if qe_idx is ei's seed-edge direction, else main list.
    // app  = update list iff DirToEdge(qe_idx) < ei  (matches C_DIR_TO_EDGE<ei).
    inline void resolve(uint32_t qe_idx, uint32_t dv,
                        const uint32_t*& base, uint32_t& blen,
                        const uint32_t*& app, uint32_t& alen) const
    {
        if (is_seed(qe_idx))
            base = mirror.UpdateList(qe_idx, dv, blen);
        else
            base = mirror.MainList(qe_idx, dv, blen);
        app = nullptr; alen = 0;
        if (mirror.DirToEdge(qe_idx) < ei)
            app = mirror.UpdateList(qe_idx, dv, alen);
    }

    // Match query vertex vs_[d] (positions 0..d-1 already in matched[]).
    uint64_t run(uint8_t d)
    {
        const uint8_t u_d = go.vs_[d];
        const uint8_t pre_qv_idx = go.bni_[go.bni_offs_[d]];
        const uint32_t pre_qe_idx = query.eidx_[go.vs_[pre_qv_idx] * QV_COUNT + u_d];
        const uint32_t pre_dv = matched[pre_qv_idx];

        const uint32_t* base; uint32_t blen;
        const uint32_t* app;  uint32_t alen;
        resolve(pre_qe_idx, pre_dv, base, blen, app, alen);

        // Support mask for (ei, u_d): a cheap superset prune that rejects most
        // invalid candidates BEFORE the injectivity scan + backward-edge binary
        // searches (mirrors the GPU's smask_read). Count-preserving.
        const uint32_t* cur_vb = mirror.SmPtr(ei, u_d);

        // HOIST: the other backward neighbors' parent vertices (matched[bni]) are
        // FIXED for the entire time we are at depth d, so resolve their candidate
        // lists ONCE here instead of per-candidate inside try_nbr. This removes
        // ~(blen+alen)*n_other redundant resolve() + binary searches per frame.
        const uint8_t oth_s = (uint8_t)(go.bni_offs_[d] + 1);
        const uint8_t oth_e = go.bni_offs_[d + 1];
        const uint8_t n_other = oth_e - oth_s;
        const uint32_t* b2[MAX_VCOUNT]; uint32_t bl2[MAX_VCOUNT];
        const uint32_t* a2[MAX_VCOUNT]; uint32_t al2[MAX_VCOUNT];
        for (uint8_t k = 0; k < n_other; k++) {
            const uint8_t bni = go.bni_[oth_s + k];
            const uint32_t qe2 = query.eidx_[go.vs_[bni] * QV_COUNT + u_d];
            resolve(qe2, matched[bni], b2[k], bl2[k], a2[k], al2[k]);
        }

        uint64_t cnt = 0;
        auto try_nbr = [&](uint32_t nbr) {
            // support-mask prune (1 word read + bit test) before the costly checks
            if (cur_vb && !(cur_vb[nbr >> 5] & (1u << (nbr & 31)))) return;
            // injectivity vs already-matched (d<=12; L1-resident linear scan beats a bitmap)
            for (uint8_t i = 0; i < d; i++)
                if (matched[i] == nbr) return;
            // other backward neighbors (binary search over the pre-resolved lists)
            for (uint8_t k = 0; k < n_other; k++)
                if (!list_contains(b2[k], bl2[k], a2[k], al2[k], nbr)) return;
            if (d == (uint8_t)(QV_COUNT - 1))
                cnt++;
            else {
                matched[d] = nbr;
                cnt += run((uint8_t)(d + 1));
            }
        };
        for (uint32_t k = 0; k < blen; k++) try_nbr(base[k]);
        for (uint32_t k = 0; k < alen; k++) try_nbr(app[k]);
        return cnt;
    }
};

} // namespace

uint64_t cpuExtendBFSAllBitCount(const uint32_t* frontier, uint64_t res_size,
                                 uint8_t cur_depth,
                                 const CPUIndexMirror& mirror,
                                 const QueryGraph& query, const Plan& plan)
{
    if (res_size == 0) return 0;
    // Precompute seed-edge directed indices per query edge ei (from indexing_orders_).
    // seed dirs = both directions of (indexing_orders_[ei].vs_[0], vs_[1]).
    uint32_t seed_a[MAX_ECOUNT], seed_b[MAX_ECOUNT];
    for (uint8_t e = 0; e < QE_COUNT; e++) {
        const auto& io = plan.indexing_orders_[e];
        uint8_t a = io.vs_[0], b = io.vs_[1];
        seed_a[e] = query.eidx_[a * QV_COUNT + b];
        seed_b[e] = query.eidx_[b * QV_COUNT + a];
    }

    DFSRunner runner(mirror, query, plan);
    uint64_t total = 0;
    const uint8_t D = cur_depth;
    for (uint64_t r = 0; r < res_size; r++) {
        const uint32_t* row = frontier + (uint64_t)r * D;
        uint32_t w0 = row[0];
        runner.ei = (uint8_t)(w0 >> 27);
        runner.matched[0] = w0 & 0x07FFFFFFu;
        runner.matched[1] = row[1];
        for (uint8_t i = 2; i < D; i++) runner.matched[i] = row[i];
        runner.seed_dir_a = (runner.ei < QE_COUNT) ? seed_a[runner.ei] : UINT32_MAX;
        runner.seed_dir_b = (runner.ei < QE_COUNT) ? seed_b[runner.ei] : UINT32_MAX;

        if (D >= QV_COUNT) { total++; continue; }  // frontier already a full embedding
        total += runner.run(D);
    }
    return total;
}

#ifdef ENABLE_CPU_DFS
uint64_t cpuExtendBFSAllBitCountParallel(const uint32_t* frontier, uint64_t res_size,
                                         uint8_t cur_depth,
                                         const CPUIndexMirror& mirror,
                                         const QueryGraph& query, const Plan& plan,
                                         tbb::task_arena& arena)
{
    if (res_size == 0) return 0;
    // Precompute seed-edge directed indices per query edge (read-only, shared).
    uint32_t seed_a[MAX_ECOUNT], seed_b[MAX_ECOUNT];
    for (uint8_t e = 0; e < QE_COUNT; e++) {
        const auto& io = plan.indexing_orders_[e];
        uint8_t a = io.vs_[0], b = io.vs_[1];
        seed_a[e] = query.eidx_[a * QV_COUNT + b];
        seed_b[e] = query.eidx_[b * QV_COUNT + a];
    }
    const uint8_t D = cur_depth;

    uint64_t total = 0;
    arena.execute([&] {
        // grain=1 + simple_partitioner: per-row cost varies by orders of magnitude
        // (some rows explode into billions of embeddings). A coarse grain lets one
        // heavy row stall a whole chunk while other threads idle; grain=1 lets TBB
        // steal single rows for balanced load.
        total = tbb::parallel_reduce(
            tbb::blocked_range<uint64_t>(0, res_size, 1), 0ull,
            [&](const tbb::blocked_range<uint64_t>& r, uint64_t init) {
                DFSRunner runner(mirror, query, plan);
                for (uint64_t row = r.begin(); row < r.end(); row++) {
                    const uint32_t* p = frontier + row * D;
                    uint32_t w0 = p[0];
                    runner.ei = (uint8_t)(w0 >> 27);
                    runner.matched[0] = w0 & 0x07FFFFFFu;
                    runner.matched[1] = p[1];
                    for (uint8_t i = 2; i < D; i++) runner.matched[i] = p[i];
                    runner.seed_dir_a = (runner.ei < QE_COUNT) ? seed_a[runner.ei] : UINT32_MAX;
                    runner.seed_dir_b = (runner.ei < QE_COUNT) ? seed_b[runner.ei] : UINT32_MAX;
                    if (D >= QV_COUNT) { init++; continue; }
                    init += runner.run(D);
                }
                return init;
            },
            [](uint64_t a, uint64_t b) { return a + b; });
    });
    return total;
}
#endif
