#ifndef INDEX_CPU_DFS_H
#define INDEX_CPU_DFS_H

#include <cstdint>
#include "index/cpu_index_mirror.h"
#ifdef ENABLE_CPU_DFS
#include <tbb/task_arena.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

class QueryGraph;
class Plan;

// ============================================================================
// CPU port of extendBFSAllBit's final-DFS pass (kernels/enumeration.cu:812).
//
// Counts complete embeddings by extending each frontier row (a partial
// embedding produced by the GPU's BFS steps) from cur_depth to QV_COUNT.
//
// Correctness: does NOT consult flat_support_masks_ — the support mask is a
// pure prune. Acceptance is fully determined by:
//   - first backward edge: candidate is enumerated from the first-BN parent's
//     neighbor list (base ++ update, per the SetupLocalIndex aliasing +
//     C_DIR_TO_EDGE<ei visibility rule);
//   - other backward edges: binary-searched in each parent's list;
//   - injectivity vs already-matched vertices.
// This is exactly the GPU kernel's logic (minus the mask prune), so the count
// matches the GPU bit-for-bit.
//
// Phase A: single-threaded, processes ALL frontier rows (for correctness
// cross-check against GPU-only num_matches). Phase B will parallelize + split.
// ============================================================================
uint64_t cpuExtendBFSAllBitCount(const uint32_t* frontier, uint64_t res_size,
                                 uint8_t cur_depth,
                                 const CPUIndexMirror& mirror,
                                 const QueryGraph& query, const Plan& plan);

#ifdef ENABLE_CPU_DFS
// Multi-threaded version (Phase B). Runs the per-row DFS on the given TBB arena
// (NUMA-2 bound) via parallel_reduce. Same count as the single-thread version.
uint64_t cpuExtendBFSAllBitCountParallel(const uint32_t* frontier, uint64_t res_size,
                                         uint8_t cur_depth,
                                         const CPUIndexMirror& mirror,
                                         const QueryGraph& query, const Plan& plan,
                                         tbb::task_arena& arena);
#endif

#endif // INDEX_CPU_DFS_H
