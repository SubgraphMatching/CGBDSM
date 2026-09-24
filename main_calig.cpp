#include <cstring>
#include <string>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <tbb/task_arena.h>
#include <tbb/info.h>
#include "utils/numa_helpers.h"

// Pick a TBB NUMA id TBB actually knows about (respects numactl --cpunodebind).
// Returns `prefer` if available (and != avoid), else the first available node
// that != avoid, else any available node, else -1 (automatic).
static int resolveTbbNumaNode(int prefer, int avoid = -1) {
    auto nodes = tbb::info::numa_nodes();
    if (nodes.empty()) return -1;
    for (auto n : nodes) if ((int)n == prefer && (int)n != avoid) return prefer;
    for (auto n : nodes) if ((int)n != avoid) return (int)n;
    return (int)nodes.front();
}

#include "utils/constants.h"
#include "utils/cuda_helpers.h"
#include "graph/graph.h"
#include "graph/graph_gpu.h"
#include "graph/plan.h"
#include "index/calig.h"
#ifdef USE_GPMA
#include "index/calig_helper_gpma.h"
#else
#define USE_GAMMA
#include "index/calig_helper_gamma.h"
#endif
#ifdef ENABLE_CPU_DFS
#include "index/cpu_dfs.h"
#endif

int main(int argc, char *argv[])
{
    std::string query_path = argv[1];
    std::string data_path = argv[2];
    std::string update_path = argv[3];

    int num_numa_nodes = getNumNumaNodes();
    int total_cores = std::thread::hardware_concurrency();
    int cores_per_node = total_cores / std::max(num_numa_nodes, 1);

    std::cout << "[Init] NUMA nodes: " << num_numa_nodes
              << ", total cores: " << total_cores
              << ", cores_per_node: " << cores_per_node << "\n"
              << "[Pipeline] 2-Stage: S1 → NUMA 0\n";

    // Bind Stage-1 to a NUMA node TBB actually knows (respects numactl cpunodebind).
    // Prefers 0; falls back to the first available node; automatic if none.
    int s1_numa = resolveTbbNumaNode(0);
    auto s1_c = tbb::task_arena::constraints{}.set_max_concurrency(cores_per_node);
    if (s1_numa >= 0) s1_c.set_numa_id(s1_numa);
    tbb::task_arena s1_arena(s1_c);

    cudaSetDevice(atoi(argv[4]));
    uint32_t batch_size;
    if (argc > 5)
        batch_size = atoi(argv[5]);
    else
        batch_size = UINT32_MAX;
    TIME_INIT();
    LTIME_INIT();
    RTIME_INIT();

    uint32_t cardinalities[MAX_ECOUNT * 2] = {0};
    float avg_degrees[MAX_ECOUNT * 2] = {0};
    CaLiG *calig_ptr = nullptr;
    uint32_t num_batches = 0, num_updates = 0;
    QueryGraph query_graph;
    Plan *plan_ptr = nullptr;
    RelationsGPU *index_gpu_ptr = nullptr, *update_index_ptr = nullptr, *local_index_ptr = nullptr;
#ifdef USE_GAMMA
    CaLiGHelperGamma *calig_helper_ptr = nullptr;
#else
    CaLiGHelper *calig_helper_ptr = nullptr;
#endif
    GPUGraphLoader *gpu_loader_ptr = nullptr;

    s1_arena.execute([&]()
                     {
        std::cout << "-----------Symbi Loading graphs ------------" << std::endl;
        calig_ptr = new CaLiG();
        calig_ptr->inputQ(query_path);
        if (data_path.compare(data_path.length() - 6, 6, ".graph") == 0)
        {
            calig_ptr->inputG(data_path);
            calig_ptr->inputUpdate(update_path, 0);
        }
        else
        {
            calig_ptr->inputGBin(data_path);
        }

        num_updates = calig_ptr->getUpdateSize();
        num_batches = (batch_size >= num_updates) ? 1 : (num_updates + batch_size - 1) / batch_size;
        std::cout << "Total updates: " << num_updates << ", batch_size: " << batch_size << ", num_batches: " << num_batches << '\n';
        calig_ptr->constructCand(num_batches);
        calig_ptr->staticFilter(); });

    s1_arena.execute([&]()
                     {
        std::cout << "----------- CPU Load Graph ------------\n";
        plan_ptr = new Plan(query_graph);

        CPUGraphLoader cpu_loader(query_path, query_graph);
        cpu_loader.SetQueryMeta();
        plan_ptr->GenerateIndexingOrders_v2();

        std::cout << "----------- GPU Load Graph ------------\n";
        index_gpu_ptr = new RelationsGPU();
        update_index_ptr = new RelationsGPU();
        local_index_ptr = new RelationsGPU();
        gpu_loader_ptr = new GPUGraphLoader(cpu_loader, query_graph, *plan_ptr);
#ifdef USE_GAMMA
        calig_helper_ptr = new CaLiGHelperGamma(calig_ptr, index_gpu_ptr, update_index_ptr,
                                                 local_index_ptr, plan_ptr, num_batches,
                                                 &gpu_loader_ptr->getNbrMemPool());
#else
        calig_helper_ptr = new CaLiGHelper(calig_ptr, index_gpu_ptr, update_index_ptr,
                                           local_index_ptr, plan_ptr, num_batches);
#endif
        gpu_loader_ptr->LoadQuery();

        std::cout << "----------- GPU Memory Allocation ------------\n";
        gpu_loader_ptr->AllocRelations(calig_ptr->G.size());

#ifdef ENABLE_CPU_DFS
        // Create the CPU DFS arena EARLY (before ConvertGlobalIndex) and route the
        // mirror's parallel work onto it, so the mirror data is first-touched on the
        // CPU DFS NUMA node (local reads during DFS). The arena avoids the Stage-1
        // node (s1_numa) so CPU DFS gets dedicated cores (e.g. NUMA-1 when S1=NUMA-0).
        {
            int cpu_numa = 2;
            if (const char* e = getenv("CGCSM_CPU_DFS_NUMA")) cpu_numa = atoi(e);
            gpu_loader_ptr->InitCPUDfs(calig_helper_ptr->GetCPUMirror(), cpu_numa, s1_numa);
            calig_helper_ptr->GetCPUMirror()->SetArena(gpu_loader_ptr->GetCPUDfsArena());
        }
#endif

        std::cout << "----------- Query Plan Generation ------------\n";
        calig_helper_ptr->ConvertGlobalIndex(cardinalities, avg_degrees);
        plan_ptr->GenerateMatchingOrders_v2(cardinalities, avg_degrees);
// #ifdef USE_MERGED_MATCHING // 这个地方存疑，因为Rebuild的时候只读取Rebuild_V_config，这个存疑存疑！！！可能是正确的
//         // Force-rebuild u0/u1 1-hop query neighbors so their support masks cover u0/u1
//         // adjacency (mask then guarantees candidate~v0/v1), enabling sound relax of the
//         // u0/u1 connection check. Cheaper than rebuilding the whole graph.
//         {
//             uint8_t gu0 = plan_ptr->global_order_.vs_[0];
//             uint8_t gu1 = plan_ptr->global_order_.vs_[1];
//             for (uint8_t e = 0; e < plan_ptr->query_.ecount_; ++e)
//                 for (uint8_t root : {gu0, gu1})
//                     for (auto& nbp : plan_ptr->query_.nbrs_[root]) {
//                         uint8_t nb = static_cast<uint8_t>(nbp.first);
//                         plan_ptr->rebuild_v_flags_[e][nb] = 1;
//                         plan_ptr->rebuild_flags_[e][nb * MAX_VCOUNT + root] = 1;
//                         plan_ptr->rebuild_flags_[e][root * MAX_VCOUNT + nb] = 1;
//                     }
//         }
// #endif
        gpu_loader_ptr->LoadPlan();
        // Let DetectConflict read qe_list_ directly so its ei numbering matches the GPU.
        // calig_ptr->setQueryGraph(plan_ptr->query_);

        plan_ptr->PrintOrders();

        std::cout << "--------- Incremental Matching (2-Stage Pipeline) --------\n";
        gpu_loader_ptr->AllocOnline(); });

    TIME_START();

    unsigned long long int num_positive_matches = 0ull;
    double s1_total_updateIndex = 0, s1_total_ConstructUpdate = 0;
    double s1_total_PrepareUpdateAll = 0, s1_total_PrepareUpdateIndex = 0;
    double gpu_total_TransferUpdateAll = 0;
    double gpu_total_TransferUpdateIndex = 0, gpu_total_UpdateGlobalIndex = 0;
    double gpu_total_BuildLocalIndex = 0, gpu_total_Matching = 0;
    double total_s1_wait = 0;

    if (num_batches == 0)
    {
        std::cout << "No updates to process.\n";
        return 0;
    }

// #ifdef USE_MERGED_MATCHING
//     uint64_t dc_reached_tot = 0, dc_conflicted_tot = 0, dc_batches = 0;
//     double dc_total_ms = 0;
//     const char* env_relax = getenv("CGCSM_RELAX_GATE");
//     const bool enable_relax = env_relax && atoi(env_relax);
//     if (env_relax) std::cout << "[RelaxGate] enabled=" << enable_relax << "\n";
//     std::vector<std::vector<uint8_t>> cf_pb(num_batches);
// #endif

    std::mutex pipe_mtx;
    std::condition_variable cv_s1_done;
    int s1_batch = -1;
    bool pipeline_end = false;

    struct S1Timing
    {
        double updateIndex_ms, constructUpdate_ms, prepareAll_ms, prepareIndex_ms;
    };
    std::vector<S1Timing> s1_timings(num_batches);

    // ---- Stage 1 persistent thread (NUMA 0) ----
    // Combines former S1 (updateIndex + ConstructUpdate) + S2 (PrepareUpdateAll + PrepareUpdateIndex)
    auto stage1_func = [&]()
    {
        s1_arena.execute([&]()
                         {
            for (uint32_t b = 0; b < num_batches; b++) {
                uint32_t start = b * batch_size;
                uint32_t count = std::min(batch_size, calig_ptr->getUpdateSize() - start);

                auto t0 = std::chrono::high_resolution_clock::now();
                calig_ptr->updateIndex(start, count, b);
                auto t1 = std::chrono::high_resolution_clock::now();

                calig_ptr->ConstructUpdate(start, count);
                auto t2 = std::chrono::high_resolution_clock::now();

                calig_ptr->mergeDeltas(b);
                auto t2b = std::chrono::high_resolution_clock::now();

// #ifdef USE_MERGED_MATCHING
//                 {
//                     auto dc0 = std::chrono::high_resolution_clock::now();
//                     calig_ptr->DetectConflict(start, count);
//                     auto dc1 = std::chrono::high_resolution_clock::now();
//                     dc_total_ms += std::chrono::duration<double, std::milli>(dc1 - dc0).count();
//                     dc_reached_tot += calig_ptr->dc_reached_;
//                     dc_conflicted_tot += calig_ptr->dc_conflicted_;
//                     ++dc_batches;
//                 }
// #endif

                // Former Stage 2 work (now sequential, no snapshot needed)
                calig_helper_ptr->PrepareUpdateAll(b);
                auto t3 = std::chrono::high_resolution_clock::now();

                calig_helper_ptr->PrepareUpdateIndex(b);
                auto t4 = std::chrono::high_resolution_clock::now();

// #ifdef USE_MERGED_MATCHING
//                 if (enable_relax)
//                     cf_pb[b] = calig_helper_ptr->BuildCompressedConflictFree2D();
// #endif

                s1_timings[b].updateIndex_ms = static_cast<double>(std::chrono::duration<double, std::milli>(t1 - t0).count());
                s1_timings[b].constructUpdate_ms = static_cast<double>(std::chrono::duration<double, std::milli>(t2 - t1).count())
                                                  + static_cast<double>(std::chrono::duration<double, std::milli>(t2b - t2).count());
                s1_timings[b].prepareAll_ms = static_cast<double>(std::chrono::duration<double, std::milli>(t3 - t2b).count());
                s1_timings[b].prepareIndex_ms = static_cast<double>(std::chrono::duration<double, std::milli>(t4 - t3).count());

                std::lock_guard<std::mutex> lk(pipe_mtx);
                s1_batch = b;
                cv_s1_done.notify_one();
            }
            {
                std::lock_guard<std::mutex> lk(pipe_mtx);
                pipeline_end = true;
            }
            cv_s1_done.notify_one(); });
    };

    std::thread s1_thread(stage1_func);

    // ---- Stage 2: Main thread loop (GPU Transfer + Matching) ----
// #if RELAX_DEBUG
//     cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32 * 1024 * 1024);
// #endif
    for (uint32_t b = 0; b < num_batches; b++)
    {
        uint32_t start = b * batch_size;
        uint32_t count = std::min(batch_size, calig_ptr->getUpdateSize() - start);
        {
            auto wt0 = std::chrono::high_resolution_clock::now();
            std::unique_lock<std::mutex> lk(pipe_mtx);
            cv_s1_done.wait(lk, [&]
                            { return s1_batch >= (int)b || pipeline_end; });
            auto wt1 = std::chrono::high_resolution_clock::now();
            total_s1_wait += static_cast<double>(std::chrono::duration<double, std::milli>(wt1 - wt0).count());
        }

        // Transfer
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            calig_helper_ptr->TransferUpdateAll(b);
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_total_TransferUpdateAll += static_cast<double>(std::chrono::duration<double, std::milli>(t1 - t0).count());

            if (!calig_helper_ptr->TransferUpdateIndex(b))
                continue;
            // calig_helper_ptr->PrintUpdateIndexSizes(b);
            auto t2 = std::chrono::high_resolution_clock::now();
            gpu_total_TransferUpdateIndex += static_cast<double>(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }

        // GPU Match
        {
#ifdef USE_VALID_BITS
            auto t_0 = std::chrono::high_resolution_clock::now();
            gpu_loader_ptr->BuildLocalIndexBitAll(*index_gpu_ptr, *update_index_ptr, avg_degrees);
#ifdef ENABLE_CPU_DFS
            // D2H the support mask now (on the side stream); it overlaps the GPU
            // BFS steps inside MatchingBitAll and is ready before the CPU DFS.
            gpu_loader_ptr->SyncCPUMirrorMasks();
#endif
            auto t_1 = std::chrono::high_resolution_clock::now();
            gpu_total_BuildLocalIndex += static_cast<double>(std::chrono::duration<double, std::milli>(t_1 - t_0).count());
            bool edge_ok[MAX_ECOUNT];
            memcpy(edge_ok, calig_helper_ptr->GetEdgeOk(b), sizeof(bool) * QE_COUNT);
#endif
#ifdef USE_MERGED_MATCHING
            // if (enable_relax && !cf_pb[b].empty())
            //     gpu_loader_ptr->UploadConflictFree(cf_pb[b].data(), static_cast<uint32_t>(cf_pb[b].size()));
            RTIME_START();
            gpu_loader_ptr->MatchingBitAll(*index_gpu_ptr, *update_index_ptr, edge_ok, num_positive_matches);
            RTIME_END();
            gpu_total_Matching += static_cast<double>(std::chrono::duration<double, std::milli>(rdiff).count());
#ifndef USE_GPMA
#ifdef USE_GAMMA
            // GAMMA: batch all UpdateGlobalIndex into one pass
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                calig_helper_ptr->BatchUpdateGlobalIndex(b);
                auto t1 = std::chrono::high_resolution_clock::now();
                gpu_total_UpdateGlobalIndex += static_cast<double>(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
#else
            // After matching, update index_gpu for next batch
            for (uint8_t i = 0u; i < QE_COUNT; i++)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                calig_helper_ptr->UpdateGlobalIndex(b, i);
                auto t1 = std::chrono::high_resolution_clock::now();
                gpu_total_UpdateGlobalIndex += static_cast<double>(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
#endif
#endif
#else
            for (uint8_t i = 0u; i < QE_COUNT; i++)
            {
#ifndef USE_GPMA
                auto t0 = std::chrono::high_resolution_clock::now();
                calig_helper_ptr->UpdateGlobalIndex(b, i);
                auto t1 = std::chrono::high_resolution_clock::now();
                gpu_total_UpdateGlobalIndex += static_cast<double>(std::chrono::duration<double, std::milli>(t1 - t0).count());
#endif
#ifdef USE_VALID_BITS
                if (!edge_ok[i])
                    continue;
                gpu_loader_ptr->SetupLocalIndex(*index_gpu_ptr, *local_index_ptr, *update_index_ptr, i);

                RTIME_START();
                gpu_loader_ptr->MatchingBit(*local_index_ptr, i, num_positive_matches);
#else
                t0 = std::chrono::high_resolution_clock::now();
                bool ok = gpu_loader_ptr->BuildLocalIndex(*index_gpu_ptr, *local_index_ptr, *update_index_ptr, i, avg_degrees);
                t1 = std::chrono::high_resolution_clock::now();
                gpu_total_BuildLocalIndex += static_cast<double>(std::chrono::duration<double, std::milli>(t1 - t0).count());
                if (!ok)
                    continue;

                RTIME_START();
                gpu_loader_ptr->Matching(*local_index_ptr, i, num_positive_matches);
#endif
                RTIME_END();
                gpu_total_Matching += static_cast<double>(std::chrono::duration<double, std::milli>(rdiff).count());
            }
#endif
        }

        // std::cout << "Num Positive Matches: " << num_positive_matches << '\n';
    }

    s1_thread.join();

    TIME_END();
    PRINT_LOCAL_TIME("Incremental Matching (2-Stage Pipeline)");

    // Accumulate S1 timings from thread
    for (uint32_t b = 0; b < num_batches; b++)
    {
        s1_total_updateIndex += s1_timings[b].updateIndex_ms;
        s1_total_ConstructUpdate += s1_timings[b].constructUpdate_ms;
        s1_total_PrepareUpdateAll += s1_timings[b].prepareAll_ms;
        s1_total_PrepareUpdateIndex += s1_timings[b].prepareIndex_ms;
    }

    std::cout << "Num Positive Matches: " << num_positive_matches << '\n';
#ifdef ENABLE_CPU_DFS
    std::cout << "[CPU_DFS] final EMA ratio = " << gpu_loader_ptr->GetCPUDfsRatio() << '\n';
    std::cout << "[CPU_DFS] mirror build (init) = " << calig_helper_ptr->GetCPUMirrorBuildMs() << " ms\n";
    std::cout << "[CPU_DFS] mirror merge  ins=" << calig_helper_ptr->GetCPUMirrorInsMs()
              << " ms  del=" << calig_helper_ptr->GetCPUMirrorDelMs()
              << " ms  rebuild=" << calig_helper_ptr->GetCPUMirrorRebuildMs()
              << " ms  batch=" << calig_helper_ptr->GetCPUMirrorBatchMs() << " ms\n";
#endif

// #if RELAX_DEBUG
// #ifdef USE_MERGED_MATCHING
//     gpu_loader_ptr->RelaxDbgPrintSkipFail();
// #endif
// #endif

    std::cout << "\n========== 2-Stage Pipeline Profiling Summary ==========\n";
    std::cout << "Total batches: " << num_batches << "\n";
    std::cout << "\n--- Stage 1 (CPU: updateIndex + ConstructUpdate + PrepareUpdate) ---\n";
    std::cout << "[S1] updateIndex:        " << s1_total_updateIndex << " ms\n";
    std::cout << "[S1] ConstructUpdate:    " << s1_total_ConstructUpdate << " ms\n";
    std::cout << "[S1] PrepareUpdateAll:   " << s1_total_PrepareUpdateAll << " ms\n";
    std::cout << "[S1] PrepareUpdateIndex: " << s1_total_PrepareUpdateIndex << " ms\n";
    double s1_total = s1_total_updateIndex + s1_total_ConstructUpdate + s1_total_PrepareUpdateAll + s1_total_PrepareUpdateIndex;
    std::cout << "[S1] TOTAL:              " << s1_total << " ms\n";
    std::cout << "\n--- Stage 2 (GPU: Transfer + Match) ---\n";
    std::cout << "[S2] TransferUpdateAll:  " << gpu_total_TransferUpdateAll << " ms\n";
    std::cout << "[S2] TransferUpdateIdx:  " << gpu_total_TransferUpdateIndex << " ms\n";
    std::cout << "[S2] UpdateGlobalIndex:  " << gpu_total_UpdateGlobalIndex << " ms\n";
    std::cout << "[S2] BuildLocalIndex:    " << gpu_total_BuildLocalIndex << " ms\n";
    std::cout << "[S2] Matching:           " << gpu_total_Matching << " ms\n";
    double s2_total = gpu_total_TransferUpdateAll + gpu_total_TransferUpdateIndex + gpu_total_UpdateGlobalIndex + gpu_total_BuildLocalIndex + gpu_total_Matching;
    std::cout << "[S2] TOTAL:              " << s2_total << " ms\n";
    std::cout << "\n--- Sync Waits ---\n";
    std::cout << "[Sync] S2 wait for S1:   " << total_s1_wait << " ms\n";
    std::cout << "\n--- Bottleneck = max(S1, S2) ---\n";
    std::cout << "S1 avg/batch: " << s1_total / (num_batches > 0 ? num_batches : 1) << " ms\n";
    std::cout << "S2 avg/batch: " << s2_total / (num_batches > 0 ? num_batches : 1) << " ms\n";
    std::cout << "===================================================\n";

// #ifdef USE_MERGED_MATCHING
//     std::cout << "\n--- Conflict Detection (CPU 1-hop, per pattern edge) ---\n";
//     std::cout << "[DC] time:        " << dc_total_ms << " ms  (batches=" << dc_batches << ")\n";
//     std::cout << "[DC] reached(candidates): " << dc_reached_tot
//               << "  conflicted: " << dc_conflicted_tot << "\n";
// #endif

    delete calig_helper_ptr, gpu_loader_ptr;
    delete index_gpu_ptr, update_index_ptr;
    delete local_index_ptr, plan_ptr, calig_ptr;
}
