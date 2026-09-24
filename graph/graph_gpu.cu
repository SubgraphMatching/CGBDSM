#include <iostream>
#include <chrono>
#include <cmath>
#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/constants.h"
#include "utils/globals.h"

#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "graph/graph.h"
#include "graph/graph_gpu.h"

#include "kernels/indexing.h"
#include "kernels/enumeration.h"
#include "kernels/enumeration_balance.h"
#include "kernels/cartesian_product.h"
#ifdef ENABLE_CPU_DFS
#include "index/cpu_dfs.h"
#include <chrono>
#include <tbb/info.h>

// Pick a TBB NUMA id TBB actually knows about (respects numactl --cpunodebind).
// Avoids `avoid` (the Stage-1 node) so CPU DFS gets dedicated cores.
static int resolveTbbNumaNode(int prefer, int avoid) {
    auto nodes = tbb::info::numa_nodes();
    if (nodes.empty()) return -1;
    for (auto n : nodes) if ((int)n == prefer && (int)n != avoid) return prefer;
    for (auto n : nodes) if ((int)n != avoid) return (int)n;
    return (int)nodes.front();
}
#endif


__constant__ OrderPerEdge C_INDEXING_ORDERS[MAX_ECOUNT];
__constant__ OrderPerEdge C_ORDERS[MAX_ECOUNT];
__constant__ IndexingOrderExt C_INDEXING_ORDERS_EXT[MAX_ECOUNT];
__constant__ ValidBits C_VALID_BITS;
__constant__ uint8_t C_DIR_TO_EDGE[MAX_ECOUNT * 2];
__constant__ uint8_t C_REBUILD_V_FLAGS[MAX_ECOUNT * MAX_VCOUNT];
__constant__ float C_AVG_DEGREES[MAX_ECOUNT * 2];

#ifdef USE_MERGED_MATCHING
__constant__ OrderPerEdge C_GLOBAL_ORDER;
__constant__ uint8_t C_GLOBAL_CP_INFO[MAX_VCOUNT];
#endif

RelationsGPU::RelationsGPU()
: nbrs_()
, capability_()
, sizes_()
{}

CandidatesGPU::CandidatesGPU()
: candidate_bits_()
{}

GPUGraphLoader::GPUGraphLoader(
    const CPUGraphLoader& cpu_loader, 
    const QueryGraph& query,
    const Plan& plan)
: query_(query)
, plan_(plan)

, d_temp_storage_(NULL)
, temp_storage_bytes_(0ul)
, temp_storage_capability_(0ul)
, cand_flag_(NULL)
, cand_flag_capability_(0u)
, d_new_cand_count_{NULL, NULL}
, temp_tries_()
, temp_tries_capability_{{0u,0u,0u}, {0u,0u,0u}}
, helper_relation_{NULL, NULL}
, helper_relation_capability_{0u}
, local_nbr_()
, local_nbr_capability_{0u}
, cum_bn_()
, valid_bits_()
, valid_bits_capability_()

, res_(0ul)
, res_size_(0ul)
, new_res_(0ul)
, new_res_size_(NULL)
, h_new_res_size_(0ul)
, h_max_new_res_size_(0ul)
, cur_depth_(0u)
, new_depth_(0u)

, nbr_mem_pool_()
, res_queue_()
, res_size_cartesian_product_(NULL)
, max_res_size_cartesian_product_(NULL)
{
    nbr_mem_pool_.Alloc(NBR_SPACE);
    cudaErrorCheck(cudaMalloc(&max_res_size_cartesian_product_, sizeof(unsigned long)));

    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[0], sizeof(uint32_t)));
    cudaErrorCheck(cudaMalloc(&d_new_cand_count_[1], sizeof(uint32_t)));

    cudaErrorCheck(cudaMalloc(&new_res_size_, sizeof(unsigned long long int)));
}

GPUGraphLoader::~GPUGraphLoader()
{
    nbr_mem_pool_.Free();

    cudaErrorCheck(cudaFree(new_res_size_));

    cudaErrorCheck(cudaFree(d_new_cand_count_[0]));
    cudaErrorCheck(cudaFree(d_new_cand_count_[1]));

    if (temp_storage_capability_ > 0u) cudaErrorCheck(cudaFree(d_temp_storage_));
    if (cand_flag_capability_ > 0u) cudaErrorCheck(cudaFree(cand_flag_));

    for (auto i = 0u; i < 2u; i++)
    {
        if (temp_tries_capability_[i].vs_capability_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].vs_));
        if (temp_tries_capability_[i].off_capability_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].offs_));
        if (temp_tries_capability_[i].es_capability_ > 0u) cudaErrorCheck(cudaFree(temp_tries_[i].nbrs_));

        if (helper_relation_capability_[i] > 0u) cudaErrorCheck(cudaFree(helper_relation_[i]));
    }

    for (auto i = 0u; i < QE_COUNT; i++)
    {
        if (local_nbr_capability_[query_.qe_eidx_[i].first] > 0u) cudaErrorCheck(cudaFree(local_nbr_[query_.qe_eidx_[i].first]));
        if (local_nbr_capability_[query_.qe_eidx_[i].second] > 0u) cudaErrorCheck(cudaFree(local_nbr_[query_.qe_eidx_[i].second]));
    }

    for (uint8_t i = 0; i < QV_COUNT; i++)
    {
        if (valid_bits_capability_[i] > 0u && valid_bits_.bits_[i])
            cudaErrorCheck(cudaFree(valid_bits_.bits_[i]));
    }

    if (flat_support_masks_tmp_size_ > 0u && flat_support_masks_tmp_)
        cudaErrorCheck(cudaFree(flat_support_masks_tmp_));
    if (d_edge_sm_tmp_ptrs_)
        cudaErrorCheck(cudaFree(d_edge_sm_tmp_ptrs_));

#ifdef USE_GLOBAL_RQ
    if (d_global_rq_) cudaErrorCheck(cudaFree(d_global_rq_));
#endif

#ifdef ENABLE_CPU_DFS
    if (cpu_dfs_arena_) { delete cpu_dfs_arena_; cpu_dfs_arena_ = nullptr; }
    if (h_frontier_buf_) { cudaErrorCheck(cudaFreeHost(h_frontier_buf_)); h_frontier_buf_ = nullptr; }
    if (cpu_dfs_stream_) cudaErrorCheck(cudaStreamDestroy(cpu_dfs_stream_));
    if (cpu_dfs_ev0_) cudaErrorCheck(cudaEventDestroy(cpu_dfs_ev0_));
    if (cpu_dfs_ev1_) cudaErrorCheck(cudaEventDestroy(cpu_dfs_ev1_));
    if (cpu_mask_stream_) cudaErrorCheck(cudaStreamDestroy(cpu_mask_stream_));
#endif
}

void GPUGraphLoader::LoadQuery()
{
    cudaErrorCheck(cudaMemcpyToSymbol(C_QV_COUNT, &QV_COUNT, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_QE_COUNT, &QE_COUNT, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_QV_OFFS, query_.qv_offs_.data(), sizeof(uint8_t) * (QV_COUNT + 1u)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NLF, query_.NLF_.data(), sizeof(uint8_t) * QE_COUNT * 2u));
    cudaErrorCheck(cudaMemcpyToSymbol(C_EIDX, query_.eidx_.data(), sizeof(uint8_t) * QV_COUNT * QV_COUNT));

    // Upload direction-to-edge mapping for visibility masking
    uint8_t h_d2e[MAX_ECOUNT * 2];
    memset(h_d2e, UINT8_MAX, sizeof(h_d2e));
    for (uint8_t e = 0; e < query_.ecount_; e++) {
        h_d2e[query_.qe_eidx_[e].first] = e;
        h_d2e[query_.qe_eidx_[e].second] = e;
    }
    cudaErrorCheck(cudaMemcpyToSymbol(C_DIR_TO_EDGE, h_d2e, sizeof(uint8_t) * QE_COUNT * 2));
}

void GPUGraphLoader::LoadPlan()
{
#ifndef USE_MERGED_MATCHING
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDERS, plan_.orders_, sizeof(OrderPerEdge) * MAX_ECOUNT));
#endif
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS, plan_.indexing_orders_, sizeof(OrderPerEdge) * MAX_ECOUNT));
    cudaErrorCheck(cudaMemcpyToSymbol(C_INDEXING_ORDERS_EXT, plan_.indexing_ext_, sizeof(IndexingOrderExt) * MAX_ECOUNT));

    // Upload rebuild_v_flags for merged kernels
    uint8_t h_rebuild[MAX_ECOUNT * MAX_VCOUNT] = {};
    for (uint8_t e = 0; e < QE_COUNT; e++)
        for (uint8_t qv = 0; qv < QV_COUNT; qv++)
            h_rebuild[e * MAX_VCOUNT + qv] = plan_.rebuild_v_flags_[e][qv] ? 1 : 0;
    cudaErrorCheck(cudaMemcpyToSymbol(C_REBUILD_V_FLAGS, h_rebuild, sizeof(uint8_t) * QE_COUNT * MAX_VCOUNT));

#ifdef USE_MERGED_MATCHING
    cudaErrorCheck(cudaMemcpyToSymbol(C_GLOBAL_ORDER, &plan_.global_order_, sizeof(OrderPerEdge)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_GLOBAL_CP_INFO, plan_.global_cartesian_product_info_.data(), sizeof(uint8_t) * MAX_VCOUNT));
#endif
}

void GPUGraphLoader::AllocRelations(uint32_t DV_COUNT_) {
    cudaErrorCheck(cudaMemcpyToSymbol(C_DV_COUNT, &DV_COUNT_, sizeof(uint32_t)));
}

void GPUGraphLoader::AllocOnline() {
    cudaErrorCheck(cudaMalloc(&res_size_cartesian_product_, SIZE_SPACE * sizeof(unsigned long)));
    res_queue_.Alloc(RES_SPACE);
    cudaErrorCheck(cudaMemcpyToSymbol(C_RES_QUEUE, &res_queue_, sizeof(CyclicQueue<uint32_t>)));
#ifdef USE_GLOBAL_RQ
    const unsigned long long max_warps = (unsigned long long)GRID_DIM * NWARP_PER_BLOCK;
    size_t rq_needed = (size_t)max_warps * (QV_COUNT - 2) * 256;
    cudaErrorCheck(cudaMalloc(&d_global_rq_, rq_needed * sizeof(uint32_t)));
    global_rq_capability_ = rq_needed;
#endif
}

void GPUGraphLoader::ReAllocValidBits() {
    if (DV_COUNT == 0u) return;
    uint32_t num_words = (DV_COUNT + 31u) / 32u;
    for (uint8_t i = 0; i < QV_COUNT; i++) {
        ReAlloc(valid_bits_.bits_[i], num_words, valid_bits_capability_[i], uint32_t);
    }
    cudaErrorCheck(cudaMemcpyToSymbol(C_VALID_BITS, &valid_bits_, sizeof(ValidBits)));
}

void GPUGraphLoader::PrintGammaMetrics(const RelationsGPU& index_gpu) {
    uint32_t num_qe = query_.ecount_; 
    uint32_t h_temp[2], *d_temp;
    cudaErrorCheck(cudaMalloc(&d_temp, sizeof(uint32_t) * 2u));

    std::cout << "\n# Candidate edges (GAMMA Style - Corrected Mapping):" << std::endl;

    for (uint32_t i = 0; i < num_qe; i++) {
        const auto& idx_uv = query_.qe_eidx_[i].first;
        const auto& idx_vu = query_.qe_eidx_[i].second;
        const auto& u = query_.qe_list_[i].first;
        const auto& v = query_.qe_list_[i].second;

        cudaErrorCheck(cudaMemset(d_temp, 0u, sizeof(uint32_t) * 2u));
        statisticIndex<<<GRID_DIM, BLOCK_DIM>>>(index_gpu, idx_uv, d_temp, d_temp + 1);
        cudaErrorCheck(cudaMemcpy(h_temp, d_temp, sizeof(uint32_t) * 2u, cudaMemcpyDeviceToHost));
        uint32_t edges_uv = h_temp[0];
        uint32_t count_u = h_temp[1];

        cudaErrorCheck(cudaMemset(d_temp, 0u, sizeof(uint32_t) * 2u));
        statisticIndex<<<GRID_DIM, BLOCK_DIM>>>(index_gpu, idx_vu, d_temp, d_temp + 1);
        cudaErrorCheck(cudaMemcpy(h_temp, d_temp, sizeof(uint32_t) * 2u, cudaMemcpyDeviceToHost));
        uint32_t edges_vu = h_temp[0];
        uint32_t count_v = h_temp[1];

        std::cout << "(" << (uint32_t)u << ", " << (uint32_t)v << "): " << edges_uv 
                  << " " << (uint32_t)u << ": " << count_u 
                  << " " << (uint32_t)v << ": " << count_v;

        if (edges_uv != edges_vu) {
            std::cout << " [Asymmetric! " << edges_uv << " vs " << edges_vu << "]";
        }
        std::cout << std::endl;
    }

    cudaErrorCheck(cudaFree(d_temp));
}

// BuildLocalIndex 理论上可以并行，rebuild_flags可以同时对QE_COUNT个边进行筛选
// cub替代为Block Reducing应该可行
// 创新性？xiao
// BuildLocalIndex可不可以在某种程度上是预先匹配呢？预先匹配的话CPU有优势啊，小规模的那种
// BuildLocalIndex是否存在冗余呢？冗余，因为需要生成QE_COUNT个LocalIndex，有没有较优的方法，转为其他的图例如时序图，没有啥好办法，合并吗？
// 刚刚想到的思路是限制搜索范围在CPU上进行初筛并且作为候选传输到GPU上，不知道可行否
bool GPUGraphLoader::BuildLocalIndex(
    const RelationsGPU& index_gpu, RelationsGPU& local_index, const RelationsGPU& update_index, const uint8_t cur_i, const float *avg_degrees
) {
    static uint32_t size_cum_bn_ = 0;
    ReAlloc(cum_bn_, DV_COUNT, size_cum_bn_, uint32_t);
    cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));
    cudaErrorCheck(cudaDeviceSynchronize());

    for (auto i = 1u; i < QV_COUNT; i++)
    {
        if (i == 1)
        {
            const auto& off = plan_.indexing_orders_[cur_i].bni_offs_[i];
            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[1];
            const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

            for (auto j = 0u; j < 2u; j++)
            {
                const auto& u = j == 0u ? u0 : u1;
                const auto& uu = j == 0u ? u1 : u0;
                const auto& index = query_.eidx_[u * QV_COUNT + uu];
                cudaErrorCheck(cudaMemcpy(local_index.nbrs_[index], update_index.nbrs_[index], (DV_COUNT + 1) * sizeof(uint32_t*), cudaMemcpyDeviceToDevice));
                cudaErrorCheck(cudaMemcpy(local_index.sizes_[index], update_index.sizes_[index], (DV_COUNT + 1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index.sizes_[query_.eidx_[u * QV_COUNT + uu]], local_index.capability_[query_.eidx_[u * QV_COUNT + uu]], DV_COUNT + 1));
                auto total_size = 0u;
                cudaErrorCheck(cudaMemcpy(&total_size, local_index.capability_[query_.eidx_[u * QV_COUNT + uu]] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if (total_size == 0u) return false;
            }
            cudaErrorCheck(cudaDeviceSynchronize());
        }
        else
        {
            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[i];
#ifndef SKIP_BUILD_LOCAL_INDEX
            if (plan_.rebuild_v_flags_[cur_i][u0])
            {
                // 1. find all candidates with at least one neighbor to match each backward neighbor
                cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));

                for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                    const auto& depth = plan_.indexing_orders_[cur_i].bni_[off];
                    const auto& first_bn_of_uu = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[
                        plan_.indexing_orders_[cur_i].bni_offs_[depth]
                    ]];
                    const auto& first_bn_of_uu_index = query_.eidx_[u1 * QV_COUNT + first_bn_of_uu];

                    const auto& backward_index = query_.eidx_[u0 * QV_COUNT + u1];
                    const auto& forward_index = query_.eidx_[u1 * QV_COUNT + u0];
                    if (avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        getLocalCandidatesBackward<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, local_index, backward_index, first_bn_of_uu_index,
                            cum_bn_, off - plan_.indexing_orders_[cur_i].bni_offs_[i]
                        );
                    }
                    else
                    {
                        getLocalCandidatesForward<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, local_index, forward_index, backward_index, first_bn_of_uu_index,
                            cum_bn_, off - plan_.indexing_orders_[cur_i].bni_offs_[i]
                        );
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());
                }

                // 2. build relations from these candidates (only for vertices v if cum_bn_[v] == # backward neighbors of u)
                for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                    const auto& depth = plan_.indexing_orders_[cur_i].bni_[off];
                    const auto& first_bn_of_uu = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[
                        plan_.indexing_orders_[cur_i].bni_offs_[depth]
                    ]];
                    const auto& first_bn_of_uu_index = query_.eidx_[u1 * QV_COUNT + first_bn_of_uu];

                    const auto& backward_index = query_.eidx_[u0 * QV_COUNT + u1];
                    const auto& forward_index = query_.eidx_[u1 * QV_COUNT + u0];
                    uint8_t work_index, reversed_index;

                    cudaErrorCheck(cudaMemset(local_index.sizes_[backward_index], 0u, sizeof(uint32_t) * DV_COUNT));
                    cudaErrorCheck(cudaMemset(local_index.sizes_[forward_index], 0u, sizeof(uint32_t) * DV_COUNT));
                    cudaErrorCheck(cudaDeviceSynchronize());

                    if (avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        // build a trie u0 -> u1
                        buildLocalRelationNew2OldCount<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, local_index, backward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i]);
                        work_index = backward_index;
                        reversed_index = forward_index;
                    }
                    else
                    {
                        // build a trie u1 -> u0
                        buildLocalRelationOld2NewCount<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, local_index, forward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i]);
                        work_index = forward_index;
                        reversed_index = backward_index;
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());

                    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index.sizes_[work_index], local_index.capability_[work_index], DV_COUNT + 1));
                    auto total_size = 0u;
                    cudaErrorCheck(cudaMemcpy(&total_size, local_index.capability_[work_index] + DV_COUNT, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                    if (total_size == 0u) return false;

                    ReAlloc(helper_relation_[0], total_size, helper_relation_capability_[0], uint32_t);
                    ReAlloc(helper_relation_[1], total_size, helper_relation_capability_[1], uint32_t);
                    ReAlloc(temp_tries_[0].vs_, total_size, temp_tries_capability_[0].vs_capability_, uint32_t);
                    ReAlloc(temp_tries_[0].offs_, total_size, temp_tries_capability_[0].off_capability_, uint32_t);

                    ReAlloc(local_nbr_[work_index], total_size, local_nbr_capability_[work_index], uint32_t*);
                    ReAlloc(local_nbr_[reversed_index], total_size, local_nbr_capability_[reversed_index], uint32_t*);
                    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[work_index], local_index.capability_[work_index], DV_COUNT, local_index.nbrs_[work_index]);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    if (avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        // build a trie u0 -> u1
                        buildLocalRelationNew2OldWrite<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, local_index, backward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i], helper_relation_[0]);
                    }
                    else
                    {
                        // build a trie u1 -> u0
                        buildLocalRelationOld2NewWrite<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, local_index, forward_index, first_bn_of_uu_index,
                            cum_bn_, plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i], helper_relation_[0]);
                    }

                    // reverse temp_tries_[0] into temp_tries_[1]
                    CUB(cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_,
                        local_nbr_[work_index], helper_relation_[1], helper_relation_[0], local_nbr_[reversed_index], total_size));

                    CUB(cub::DeviceRunLengthEncode::Encode(d_temp_storage_, temp_storage_bytes_, helper_relation_[1], temp_tries_[0].vs_, temp_tries_[0].offs_, d_new_cand_count_[0], total_size));
                    cudaErrorCheck(cudaDeviceSynchronize());
                    cudaErrorCheck(cudaMemcpy(&temp_tries_[0].vs_size_, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));

                    // map trie to relation
                    mapTrieToRelation<<<GRID_DIM, BLOCK_DIM>>>(local_index, reversed_index, temp_tries_[0].vs_, temp_tries_[0].offs_, temp_tries_[0].vs_size_);
                    cudaErrorCheck(cudaDeviceSynchronize());

                    CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_, local_index.sizes_[reversed_index], local_index.capability_[reversed_index], DV_COUNT + 1));
                    setNeighborPointers<<<GRID_DIM, BLOCK_DIM>>>(local_nbr_[reversed_index], local_index.capability_[reversed_index], DV_COUNT, local_index.nbrs_[reversed_index]);
                    cudaErrorCheck(cudaDeviceSynchronize());
                }
            }
            else
            {
#endif
                for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                    // std::cout << "skip " << static_cast<uint32_t>(u0) << ' ' << static_cast<uint32_t>(u1) << '\n';
                    for (auto j = 0u; j < 2u; j++)
                    {
                        const auto& u = j == 0u ? u0 : u1;
                        const auto& uu = j == 0u ? u1 : u0;
                        const int index = query_.eidx_[u * QV_COUNT + uu];
                        cudaErrorCheck(cudaMemcpy(local_index.nbrs_[index], index_gpu.nbrs_[index], (DV_COUNT + 1) * sizeof(uint32_t*), cudaMemcpyDeviceToDevice));
                        cudaErrorCheck(cudaMemcpy(local_index.sizes_[index], index_gpu.sizes_[index], (DV_COUNT + 1) * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

                    }
                }
                cudaErrorCheck(cudaDeviceSynchronize());
#ifndef SKIP_BUILD_LOCAL_INDEX
            }
#endif
        }
    }

    // printf("\n========== BuildLocalIndex Filtering Stats ==========\n");
    // printf("QE_COUNT=%d, MAX_ECOUNT=%d, DV_COUNT=%u\n", QE_COUNT, MAX_ECOUNT, DV_COUNT);

    // uint32_t *d_global_sums = nullptr;
    // uint32_t *d_local_sums = nullptr;
    // uint32_t d_global_sums_size = 0, d_local_sums_size = 0;
    // ReAlloc(d_global_sums, QE_COUNT * 2, d_global_sums_size, uint32_t);
    // ReAlloc(d_local_sums, QE_COUNT * 2, d_local_sums_size, uint32_t);

    // size_t temp_bytes = 0;
    // void* d_temp = nullptr;

    // for (int idx = 0; idx < QE_COUNT * 2; idx++) {
    //     if (index_gpu.sizes_[idx] == nullptr) {
    //         printf("idx=%d: index_gpu.sizes_[idx] is NULL, skipping\n", idx);
    //         d_global_sums[idx] = 0;
    //         continue;
    //     }
    //     cub::DeviceReduce::Sum(nullptr, temp_bytes, index_gpu.sizes_[idx], d_global_sums + idx, DV_COUNT);
    //     cudaErrorCheck(cudaMalloc(&d_temp, temp_bytes));
    //     cub::DeviceReduce::Sum(d_temp, temp_bytes, index_gpu.sizes_[idx], d_global_sums + idx, DV_COUNT);
    //     cudaError_t err = cudaGetLastError();
    //     cudaErrorCheck(cudaDeviceSynchronize());
    //     cudaErrorCheck(cudaFree(d_temp));
    //     d_temp = nullptr;
    // }

    // for (int idx = 0; idx < QE_COUNT * 2; idx++) {
    //     if (local_index.sizes_[idx] == nullptr) {
    //         printf("idx=%d: local_index.sizes_[idx] is NULL, skipping\n", idx);
    //         d_local_sums[idx] = 0;
    //         continue;
    //     }
    //     cub::DeviceReduce::Sum(nullptr, temp_bytes, local_index.sizes_[idx], d_local_sums + idx, DV_COUNT);
    //     cudaErrorCheck(cudaMalloc(&d_temp, temp_bytes));
    //     cub::DeviceReduce::Sum(d_temp, temp_bytes, local_index.sizes_[idx], d_local_sums + idx, DV_COUNT);
    //     cudaError_t err = cudaGetLastError();
    //     cudaErrorCheck(cudaDeviceSynchronize());
    //     cudaErrorCheck(cudaFree(d_temp));
    //     d_temp = nullptr;
    // }

    // uint32_t h_global_sums[QE_COUNT * 2];
    // uint32_t h_local_sums[QE_COUNT * 2];
    // cudaMemcpy(h_global_sums, d_global_sums, sizeof(uint32_t) * QE_COUNT * 2, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_local_sums, d_local_sums, sizeof(uint32_t) * QE_COUNT * 2, cudaMemcpyDeviceToHost);

    // uint32_t total_global = 0, total_local = 0;
    // for (int idx = 0; idx < QE_COUNT * 2; idx++) {
    //     if (h_global_sums[idx] > 0) {
    //         uint32_t filtered = h_global_sums[idx] - h_local_sums[idx];
    //         float ratio = h_global_sums[idx] > 0 ? 100.0f * filtered / h_global_sums[idx] : 0.0f;
    //         printf("edge_idx=%d: global=%u, local=%u, filtered=%u (%.2f%%)\n",
    //                idx, h_global_sums[idx], h_local_sums[idx], filtered, ratio);
    //         total_global += h_global_sums[idx];
    //         total_local += h_local_sums[idx];
    //     }
    // }
    // printf("TOTAL: global=%u, local=%u, filtered=%u (%.2f%%)\n",
    //        total_global, total_local, total_global - total_local,
    //        total_global > 0 ? 100.0f * (total_global - total_local) / total_global : 0.0f);
    // printf("====================================================\n\n");

    return true;
}

// Matching 多个 localIndex 如何合并？应该不容易合并，并发度，控制成本？呃
// 如何合并？
// localIndex 多次匹配
void GPUGraphLoader::Matching(RelationsGPU& local_index, const uint8_t i, unsigned long long int& num_matches)
{
    using Clock = std::chrono::high_resolution_clock;
    double t_step1 = 0, t_step2 = 0, t_step3 = 0, t_step4 = 0, t_step5 = 0;
    auto t_total_begin = Clock::now();

    res_queue_.Reset();
    bool oom = false;

    /************************************ Step 1: writeInitialResults ************************************/
    auto t0 = Clock::now();
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
    cudaErrorCheck(cudaDeviceSynchronize());

    write_initial_partial_results<<<GRID_DIM, BLOCK_DIM>>>(
        local_index, query_.eidx_[plan_.orders_[i].vs_[0] * QV_COUNT + plan_.orders_[i].vs_[1]],
        new_res_, new_res_size_, h_max_new_res_size_
    );
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    res_queue_.Push(h_new_res_size_ * new_depth_);
    res_ = new_res_;
    res_size_ = h_new_res_size_;
    auto t1 = Clock::now();
    t_step1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (res_size_ == 0ul) return;
    cur_depth_ = new_depth_;

    /************************************ Step 2: BFS small-result loop ************************************/
    t0 = Clock::now();
    while (res_size_ < MIN_NRESULTS_TO_GPU && QV_COUNT - cur_depth_ > 1)
    {
        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
        new_depth_ = cur_depth_ + 1;
        new_res_ = res_queue_.TryMax();
        h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
        cudaErrorCheck(cudaDeviceSynchronize());

        extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, local_index, i, cur_depth_, new_depth_, true
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        if (h_new_res_size_ >= h_max_new_res_size_)
        {
            oom = true;
            break;
        }
        else
        {
            res_queue_.Push(h_new_res_size_ * new_depth_);
            res_queue_.Pop(res_size_ * cur_depth_);
            res_ = new_res_;
            res_size_ = h_new_res_size_;
            if (res_size_ == 0ul) return;
            cur_depth_ = new_depth_;
        }
    }
    t1 = Clock::now();
    t_step2 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 3: multi-level BFS ************************************/
    t0 = Clock::now();
    if (!oom && plan_.cartesian_product_info_[i][cur_depth_] != Plan::CartesianProductType::TreeCartesianProduct && QV_COUNT - cur_depth_ > 2)
    {
        new_depth_ = cur_depth_;
        while (
            new_depth_ < QV_COUNT &&
            plan_.cartesian_product_info_[i][new_depth_] != Plan::CartesianProductType::TreeCartesianProduct &&
            plan_.cartesian_product_info_[i][new_depth_] != Plan::CartesianProductType::TreeSingle
        ) {
            new_depth_++;
        }
        if (QV_COUNT - new_depth_ > 0)
        {
            cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
            new_res_ = res_queue_.TryMax();
            h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
            cudaErrorCheck(cudaDeviceSynchronize());

            extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
                res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, local_index, i, cur_depth_, new_depth_, true
            );
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            if (h_new_res_size_ >= h_max_new_res_size_)
            {
                oom = true;
            }
            else
            {
                res_queue_.Push(h_new_res_size_ * new_depth_);
                res_queue_.Pop(res_size_ * cur_depth_);
                res_ = new_res_;
                res_size_ = h_new_res_size_;
                if (res_size_ == 0ul) return;
                cur_depth_ = new_depth_;
            }
        }
    }
    t1 = Clock::now();
    t_step3 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 4: Cartesian product analysis ************************************/
    t0 = Clock::now();
    bool enumerate_cartesian_product = plan_.cartesian_product_info_[i][cur_depth_] == Plan::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    if (enumerate_cartesian_product)
    {
        GetNumTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, local_index, i, QV_COUNT - cur_depth_
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        CUB(cub::DeviceReduce::Max(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, max_res_size_cartesian_product_, res_size_));
        cudaErrorCheck(cudaDeviceSynchronize());
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());

        unsigned long h_max, h_total;
        cudaErrorCheck(cudaMemcpy(&h_max, max_res_size_cartesian_product_, sizeof(unsigned long), cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(&h_total, res_size_cartesian_product_ + res_size_, sizeof(unsigned long), cudaMemcpyDeviceToHost));

        const float avg_res_size = (float)h_total / res_size_;
        const float ratio = h_max / avg_res_size;
        if (ratio < 50.f && avg_res_size >= MIN_NRESULTS_TO_GPU)
        {
            enumerate_cartesian_product = false;
        }
    }
    t1 = Clock::now();
    t_step4 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 5: final enumeration ************************************/
    t0 = Clock::now();
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    cudaErrorCheck(cudaDeviceSynchronize());

    if (enumerate_cartesian_product)
    {
        // cout << "enumerateCartesianProductTree" << endl;
        enumerateCartesianProductTree<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, local_index, i, new_res_size_, QV_COUNT - cur_depth_
        );
    }
    else
    {
        // cout << "extendBFSDFSRegTwo res_size_ = " << res_size_ << endl;
        extendBFSDFSRegTwo<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, 0, local_index, i, cur_depth_, QV_COUNT, false
        );
    }
    cudaErrorCheck(cudaDeviceSynchronize());

    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    num_matches += h_new_res_size_;
    t1 = Clock::now();
    t_step5 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t_total_end = Clock::now();
    double t_total = std::chrono::duration<double, std::milli>(t_total_end - t_total_begin).count();
    // cout << "[Matching order=" << (int)i << " Timing] "
    //      << "step1(writeInitial)=" << t_step1 << "ms, "
    //      << "step2(BFS_small)=" << t_step2 << "ms, "
    //      << "step3(BFS_multi)=" << t_step3 << "ms, "
    //      << "step4(cartesian_analysis)=" << t_step4 << "ms, "
    //      << "step5(enumeration)=" << t_step5 << "ms, "
    //      << "total=" << t_total << "ms" << endl;
}

// BuildLocalIndexBit: sparse valid-bit approach replacing dense local_index construction
bool GPUGraphLoader::BuildLocalIndexBit(
    const RelationsGPU& index_gpu, RelationsGPU& local_index, const RelationsGPU& update_index, const uint8_t cur_i, const float *avg_degrees
) {
    // Reset valid bits for all query vertices
    uint32_t num_words = (DV_COUNT + 31u) / 32u;
    for (uint8_t qv = 0; qv < QV_COUNT; qv++) {
        cudaErrorCheck(cudaMemset(valid_bits_.bits_[qv], 0u, sizeof(uint32_t) * num_words));
    }

    static uint32_t size_cum_bn_ = 0;
    ReAlloc(cum_bn_, DV_COUNT, size_cum_bn_, uint32_t);

    for (auto i = 1u; i < QV_COUNT; i++)
    {
        if (i == 1)
        {
            const auto& off = plan_.indexing_orders_[cur_i].bni_offs_[i];
            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[1];
            const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

            // Pointer assignment from update_index (host-side, no GPU transfer)
            for (auto j = 0u; j < 2u; j++)
            {
                const auto& u = j == 0u ? u0 : u1;
                const auto& uu = j == 0u ? u1 : u0;
                const auto& index = query_.eidx_[u * QV_COUNT + uu];
                local_index.nbrs_[index] = update_index.nbrs_[index];
                local_index.sizes_[index] = update_index.sizes_[index];
            }

            // Check emptiness using CUB Reduce::Sum (both directions)
            const auto& index_u0_u1 = query_.eidx_[u0 * QV_COUNT + u1];
            const auto& index_u1_u0 = query_.eidx_[u1 * QV_COUNT + u0];
            for (auto idx : {index_u0_u1, index_u1_u0}) {
                CUB(cub::DeviceReduce::Sum(d_temp_storage_, temp_storage_bytes_,
                    local_index.sizes_[idx], d_new_cand_count_[0], DV_COUNT));
                cudaErrorCheck(cudaDeviceSynchronize());
                uint32_t total_size = 0u;
                cudaErrorCheck(cudaMemcpy(&total_size, d_new_cand_count_[0], sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if (total_size == 0u) return false;
            }

            // Set initial valid bits from update_index for both u0 and u1
            setInitialValidBits<<<GRID_DIM, BLOCK_DIM>>>(update_index, index_u0_u1, u0);
            setInitialValidBits<<<GRID_DIM, BLOCK_DIM>>>(update_index, index_u1_u0, u1);
            cudaErrorCheck(cudaDeviceSynchronize());
        }
        else
        {
            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[i];

            // Pointer assignment: all backward edge directions from index_gpu
            for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
            {
                const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                for (auto j = 0u; j < 2u; j++)
                {
                    const auto& u = j == 0u ? u0 : u1;
                    const auto& uu = j == 0u ? u1 : u0;
                    const auto& index = query_.eidx_[u * QV_COUNT + uu];
                    local_index.nbrs_[index] = index_gpu.nbrs_[index];
                    local_index.sizes_[index] = index_gpu.sizes_[index];
                }
            }

            if (plan_.rebuild_v_flags_[cur_i][u0])
            {
                // Compute cum_bn using *Bit kernel variants
                cudaErrorCheck(cudaMemset(cum_bn_, 0u, sizeof(uint32_t) * DV_COUNT));
                auto num_bn = plan_.indexing_orders_[cur_i].bni_offs_[i + 1] - plan_.indexing_orders_[cur_i].bni_offs_[i];

                for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i]; off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++)
                {
                    const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];

                    const auto& backward_index = query_.eidx_[u0 * QV_COUNT + u1];
                    const auto& forward_index = query_.eidx_[u1 * QV_COUNT + u0];
                    if (avg_degrees[backward_index] < avg_degrees[forward_index])
                    {
                        getLocalCandidatesBackwardBit<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, backward_index, u1,
                            cum_bn_, off - plan_.indexing_orders_[cur_i].bni_offs_[i]
                        );
                    }
                    else
                    {
                        getLocalCandidatesForwardBit<<<GRID_DIM, BLOCK_DIM>>>(
                            index_gpu, forward_index, backward_index, u1,
                            cum_bn_, off - plan_.indexing_orders_[cur_i].bni_offs_[i]
                        );
                    }
                    cudaErrorCheck(cudaDeviceSynchronize());
                }

                // Set valid bits for u0 where cum_bn == num_bn
                setLocalCandidateValidBits<<<GRID_DIM, BLOCK_DIM>>>(cum_bn_, num_bn, u0);
                cudaErrorCheck(cudaDeviceSynchronize());
            }
            else
            {
                // Set valid bits from global index (no filtering needed for this vertex)
                const auto& first_bn_off = plan_.indexing_orders_[cur_i].bni_offs_[i];
                const auto& first_bn = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[first_bn_off]];
                const auto& first_bn_index = query_.eidx_[u0 * QV_COUNT + first_bn];
                setInitialValidBits<<<GRID_DIM, BLOCK_DIM>>>(index_gpu, first_bn_index, u0);
                cudaErrorCheck(cudaDeviceSynchronize());
            }
        }
    }

    return true;
}

// BuildLocalIndexBitAll: merged valid bits construction for all edges
void GPUGraphLoader::BuildLocalIndexBitAll(
    const RelationsGPU& index_gpu, const RelationsGPU& update_index,
    const float* avg_degrees
) {
    size_t dv_stride = SMASK_DV_STRIDE(DV_COUNT);
    size_t total_sm = (size_t)MAX_ECOUNT * MAX_VCOUNT * dv_stride;
    ReAlloc(flat_support_masks_, total_sm, flat_support_masks_size_, smask_t);
    if (!d_edge_sm_ptrs_) cudaErrorCheck(cudaMalloc(&d_edge_sm_ptrs_, sizeof(smask_t*) * MAX_ECOUNT * MAX_VCOUNT));
    std::vector<smask_t*> h_ep(MAX_ECOUNT * MAX_VCOUNT, nullptr);
    for (uint8_t e = 0; e < MAX_ECOUNT; e++)
        for (uint8_t ei = 0; ei < MAX_VCOUNT; ei++)
            h_ep[e * MAX_VCOUNT + ei] = flat_support_masks_ + (e * MAX_VCOUNT + ei) * dv_stride;
    cudaErrorCheck(cudaMemcpy(d_edge_sm_ptrs_, h_ep.data(), sizeof(smask_t*) * MAX_ECOUNT * MAX_VCOUNT, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemset(flat_support_masks_, 0u, sizeof(smask_t) * total_sm));

    // rebuild_config: for small bit-widths, pre-fill non-rebuild vertices with SMASK_ALL
#if SUPPORT_MASK_WIDTH <= 16
    for (uint8_t e = 0; e < QE_COUNT; e++) 
        for (uint8_t qv = 0; qv < QV_COUNT; qv++) {
            if (!plan_.rebuild_v_flags_[e][qv]) {
                size_t offset = (e * MAX_VCOUNT + qv) * dv_stride;
                cudaErrorCheck(cudaMemset(flat_support_masks_ + offset, 0xFF,
                    sizeof(smask_t) * dv_stride));
            }
        }
#endif

    dim3 grid(GRID_DIM, QE_COUNT);
#if SUPPORT_MASK_WIDTH == 1
    setInitialValidBitsAll1Bit<<<grid, BLOCK_DIM>>>(update_index, d_edge_sm_ptrs_);
#else
    setInitialValidBitsAll<<<grid, BLOCK_DIM>>>(update_index, d_edge_sm_ptrs_);
#endif

#if defined(USE_PUSH_ALL)
    // Allocate temp buffer for push-all (same layout as flat_support_masks_)
    {
        size_t total_sm_tmp = (size_t)MAX_ECOUNT * MAX_VCOUNT * dv_stride;
        ReAlloc(flat_support_masks_tmp_, total_sm_tmp, flat_support_masks_tmp_size_, smask_t);
        if (!d_edge_sm_tmp_ptrs_) cudaErrorCheck(cudaMalloc(&d_edge_sm_tmp_ptrs_, sizeof(smask_t*) * MAX_ECOUNT * MAX_VCOUNT));
        std::vector<smask_t*> h_ep_tmp(MAX_ECOUNT * MAX_VCOUNT, nullptr);
        for (uint8_t e = 0; e < MAX_ECOUNT; e++)
            for (uint8_t ei = 0; ei < MAX_VCOUNT; ei++)
                h_ep_tmp[e * MAX_VCOUNT + ei] = flat_support_masks_tmp_ + (e * MAX_VCOUNT + ei) * dv_stride;
        cudaErrorCheck(cudaMemcpy(d_edge_sm_tmp_ptrs_, h_ep_tmp.data(), sizeof(smask_t*) * MAX_ECOUNT * MAX_VCOUNT, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemset(flat_support_masks_tmp_, 0u, sizeof(smask_t) * total_sm_tmp));
    }
    for (uint8_t depth = 2; depth < QV_COUNT; depth++) {
        pushAllBackwardBNAll<<<grid, BLOCK_DIM>>>(index_gpu, update_index, depth, d_edge_sm_ptrs_, d_edge_sm_tmp_ptrs_);
        intersectAndSetBitsAll<<<grid, BLOCK_DIM>>>(depth, d_edge_sm_ptrs_, d_edge_sm_tmp_ptrs_);
    }
#elif defined(USE_PUSH_VERIFY)
    for (uint8_t depth = 2; depth < QV_COUNT; depth++) {
        pushFromFirstBNAll<<<grid, BLOCK_DIM>>>(index_gpu, update_index, depth, d_edge_sm_ptrs_);
        verifyCandidatesAll<<<grid, BLOCK_DIM>>>(index_gpu, update_index, depth, d_edge_sm_ptrs_);
    }
#else
    for (uint8_t depth = 2; depth < QV_COUNT; depth++)
#if SUPPORT_MASK_WIDTH == 1
        checkAllConstraintsAndSetBitsAll1Bit<<<grid, BLOCK_DIM>>>(index_gpu, update_index, depth, d_edge_sm_ptrs_);
#else
        checkAllConstraintsAndSetBitsAll<<<grid, BLOCK_DIM>>>(index_gpu, update_index, depth, d_edge_sm_ptrs_);
#endif
#endif

    cudaErrorCheck(cudaDeviceSynchronize());
}

// SetupLocalIndex: host-side pointer assignment for local_index (extracted from BuildLocalIndexBit)
void GPUGraphLoader::SetupLocalIndex(
    const RelationsGPU& index_gpu, RelationsGPU& local_index,
    const RelationsGPU& update_index, const uint8_t cur_i
) {
    for (auto i = 1u; i < QV_COUNT; i++) {
        if (i == 1u) {
            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[1];
            const auto& off = plan_.indexing_orders_[cur_i].bni_offs_[1];
            const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];
            for (auto j = 0u; j < 2u; j++) {
                const auto& u = j == 0u ? u0 : u1;
                const auto& uu = j == 0u ? u1 : u0;
                const auto& index = query_.eidx_[u * QV_COUNT + uu];
                local_index.nbrs_[index] = update_index.nbrs_[index];
                local_index.sizes_[index] = update_index.sizes_[index];
            }
        } else {
            const auto& u0 = plan_.indexing_orders_[cur_i].vs_[i];
            for (auto off = plan_.indexing_orders_[cur_i].bni_offs_[i];
                 off < plan_.indexing_orders_[cur_i].bni_offs_[i + 1]; off++) {
                const auto& u1 = plan_.indexing_orders_[cur_i].vs_[plan_.indexing_orders_[cur_i].bni_[off]];
                for (auto j = 0u; j < 2u; j++) {
                    const auto& u = j == 0u ? u0 : u1;
                    const auto& uu = j == 0u ? u1 : u0;
                    const auto& index = query_.eidx_[u * QV_COUNT + uu];
                    local_index.nbrs_[index] = index_gpu.nbrs_[index];
                    local_index.sizes_[index] = index_gpu.sizes_[index];
                }
            }
        }
    }
}

// SetConstantValidBits: upload per-edge valid_bits to constant memory for MatchingBit
void GPUGraphLoader::SetConstantValidBits(uint8_t edge_idx) {
    uint32_t num_words = (DV_COUNT + 31u) / 32u;
    ValidBits vb;
    memset(&vb, 0, sizeof(ValidBits));
    for (uint8_t qv = 0; qv < QV_COUNT; qv++) {
        vb.bits_[qv] = flat_edge_bits_ + ((uint32_t)edge_idx * MAX_VCOUNT + qv) * num_words;
    }
    cudaErrorCheck(cudaMemcpyToSymbol(C_VALID_BITS, &vb, sizeof(ValidBits)));
}

// MatchingBit: matching with valid bit checks
void GPUGraphLoader::MatchingBit(RelationsGPU& local_index, const uint8_t i, unsigned long long int& num_matches)
{
    using Clock = std::chrono::high_resolution_clock;
    double t_step1 = 0, t_step2 = 0, t_step3 = 0, t_step4 = 0, t_step5 = 0;
    auto t_total_begin = Clock::now();

    res_queue_.Reset();
    bool oom = false;

    /************************************ Step 1: writeInitialResults ************************************/
    auto t0 = Clock::now();
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
    cudaErrorCheck(cudaDeviceSynchronize());

    write_initial_partial_results_bit<<<GRID_DIM, BLOCK_DIM>>>(
        local_index, query_.eidx_[plan_.orders_[i].vs_[0] * QV_COUNT + plan_.orders_[i].vs_[1]],
        plan_.orders_[i].vs_[0], plan_.orders_[i].vs_[1],
        new_res_, new_res_size_, h_max_new_res_size_,
        d_edge_sm_ptrs_, i
    );
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    res_queue_.Push(h_new_res_size_ * new_depth_);
    res_ = new_res_;
    res_size_ = h_new_res_size_;
    auto t1 = Clock::now();
    t_step1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (res_size_ == 0ul) return;
    cur_depth_ = new_depth_;

    /************************************ Step 2: BFS small-result loop ************************************/
    t0 = Clock::now();
    while (res_size_ < MIN_NRESULTS_TO_GPU && QV_COUNT - cur_depth_ > 1)
    {
        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
        new_depth_ = cur_depth_ + 1;
        new_res_ = res_queue_.TryMax();
        h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
        cudaErrorCheck(cudaDeviceSynchronize());

        extendBFSDFSRegTwoBit<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, local_index, i, cur_depth_, new_depth_, true,
            d_edge_sm_ptrs_, i
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        if (h_new_res_size_ >= h_max_new_res_size_)
        {
            oom = true;
            break;
        }
        else
        {
            res_queue_.Push(h_new_res_size_ * new_depth_);
            res_queue_.Pop(res_size_ * cur_depth_);
            res_ = new_res_;
            res_size_ = h_new_res_size_;
            if (res_size_ == 0ul) return;
            cur_depth_ = new_depth_;
        }
    }
    t1 = Clock::now();
    t_step2 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 3: multi-level BFS ************************************/
    t0 = Clock::now();
    if (!oom && plan_.cartesian_product_info_[i][cur_depth_] != Plan::CartesianProductType::TreeCartesianProduct && QV_COUNT - cur_depth_ > 2)
    {
        new_depth_ = cur_depth_;
        while (
            new_depth_ < QV_COUNT &&
            plan_.cartesian_product_info_[i][new_depth_] != Plan::CartesianProductType::TreeCartesianProduct &&
            plan_.cartesian_product_info_[i][new_depth_] != Plan::CartesianProductType::TreeSingle
        ) {
            new_depth_++;
        }
        if (QV_COUNT - new_depth_ > 0)
        {
            cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
            new_res_ = res_queue_.TryMax();
            h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
            cudaErrorCheck(cudaDeviceSynchronize());

            extendBFSDFSRegTwoBit<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
                res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_, local_index, i, cur_depth_, new_depth_, true,
                d_edge_sm_ptrs_, i
            );
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            if (h_new_res_size_ >= h_max_new_res_size_)
            {
                oom = true;
            }
            else
            {
                res_queue_.Push(h_new_res_size_ * new_depth_);
                res_queue_.Pop(res_size_ * cur_depth_);
                res_ = new_res_;
                res_size_ = h_new_res_size_;
                if (res_size_ == 0ul) return;
                cur_depth_ = new_depth_;
            }
        }
    }
    t1 = Clock::now();
    t_step3 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 4: Cartesian product analysis ************************************/
    t0 = Clock::now();
    bool enumerate_cartesian_product = plan_.cartesian_product_info_[i][cur_depth_] == Plan::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    if (enumerate_cartesian_product)
    {
        GetNumTreeBit<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, local_index, i, QV_COUNT - cur_depth_,
            d_edge_sm_ptrs_, i
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        CUB(cub::DeviceReduce::Max(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, max_res_size_cartesian_product_, res_size_));
        cudaErrorCheck(cudaDeviceSynchronize());
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());

        unsigned long h_max, h_total;
        cudaErrorCheck(cudaMemcpy(&h_max, max_res_size_cartesian_product_, sizeof(unsigned long), cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(&h_total, res_size_cartesian_product_ + res_size_, sizeof(unsigned long), cudaMemcpyDeviceToHost));

        const float avg_res_size = (float)h_total / res_size_;
        const float ratio = h_max / avg_res_size;
        if (ratio < 50.f && avg_res_size >= MIN_NRESULTS_TO_GPU)
        {
            enumerate_cartesian_product = false;
        }
    }
    t1 = Clock::now();
    t_step4 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 5: final enumeration ************************************/
    t0 = Clock::now();
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    cudaErrorCheck(cudaDeviceSynchronize());

    if (enumerate_cartesian_product)
    {
        // cout << "enumerate_cartesian_product " << endl;
        enumerateCartesianProductTreeBit<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_, local_index, i, new_res_size_, QV_COUNT - cur_depth_,
            d_edge_sm_ptrs_, i
        );
    }
    else
    {
        extendBFSDFSRegTwoBit<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, 0, local_index, i, cur_depth_, QV_COUNT, false,
            d_edge_sm_ptrs_, i
        );
    }
    cudaErrorCheck(cudaDeviceSynchronize());

    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    num_matches += h_new_res_size_;
    t1 = Clock::now();
    t_step5 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t_total_end = Clock::now();
    double t_total = std::chrono::duration<double, std::milli>(t_total_end - t_total_begin).count();
    // cout << "[MatchingBit order=" << (int)i << " Timing] "
    //      << "step1(writeInitial)=" << t_step1 << "ms, "
    //      << "step2(BFS_small)=" << t_step2 << "ms, "
    //      << "step3(BFS_multi)=" << t_step3 << "ms, "
    //      << "step4(cartesian_analysis)=" << t_step4 << "ms, "
    //      << "step5(enumeration)=" << t_step5 << "ms, "
    //      << "total=" << t_total << "ms" << endl;
}

#ifdef USE_MERGED_MATCHING
void GPUGraphLoader::SetupLocalIndexAll(
    const RelationsGPU& index_gpu,
    const RelationsGPU& update_index) {
    if (!d_all_local_)
        cudaErrorCheck(cudaMalloc(&d_all_local_, sizeof(RelationsGPU) * QE_COUNT));

    std::vector<RelationsGPU> h_all(QE_COUNT);
    for (uint8_t ei = 0; ei < QE_COUNT; ei++) {
        memset(&h_all[ei], 0, sizeof(RelationsGPU));
        SetupLocalIndex(index_gpu, h_all[ei], update_index, ei);
    }
    cudaErrorCheck(cudaMemcpy(d_all_local_, h_all.data(),
        sizeof(RelationsGPU) * QE_COUNT, cudaMemcpyHostToDevice));
}

// void GPUGraphLoader::UploadConflictFree(const uint8_t* cf, uint32_t size) {
//     if (size == 0) return;
//     // Track whether d_cf_ may point at a new address this call: it changes on the
//     // first call (null -> alloc) and whenever the conflict-free buffer grows (comp_dv
//     // increases across batches). On growth we cudaFree+cudaMalloc, so the device
//     // symbol g_d_cf must be re-published, or kernels dereference freed memory.
//     bool addr_may_have_changed = false;
//     if (size > d_cf_cap_) {
//         if (d_cf_) cudaErrorCheck(cudaFree(d_cf_));
//         cudaErrorCheck(cudaMalloc(&d_cf_, sizeof(uint8_t) * size));
//         d_cf_cap_ = size;
//         addr_may_have_changed = true;
//     }
//     cudaErrorCheck(cudaMemcpy(d_cf_, cf, sizeof(uint8_t) * size, cudaMemcpyHostToDevice));
//     // (Re)publish g_d_cf on first call or after a reallocation that changed d_cf_'s address.
//     if (addr_may_have_changed || !d_cf_sym_set_) {
//         extern __device__ const uint8_t* g_d_cf;
//         const uint8_t* h_ptr = d_cf_;
//         cudaErrorCheck(cudaMemcpyToSymbol(g_d_cf, &h_ptr, sizeof(uint8_t*)));
//         d_cf_sym_set_ = true;
//     }
// }

// #if RELAX_DEBUG
// void GPUGraphLoader::RelaxDbgPrintSkipFail() {
//     extern __device__ uint64_t g_dbg_skip_fail;
//     uint64_t hf = 0;
//     cudaErrorCheck(cudaMemcpyFromSymbol(&hf, g_dbg_skip_fail, sizeof(uint64_t)));
//     std::cout << "[RELAX_DEBUG] skip_fail_total=" << hf << " (over-count=25948210)\n";
// }
// #endif

#ifdef ENABLE_CPU_DFS
// Phase B: set up the host mirror + a NUMA-bound TBB arena for CPU DFS.
void GPUGraphLoader::InitCPUDfs(CPUIndexMirror* mirror, int numa_node, int avoid_node) {
    cpu_mirror_ = mirror;
    if (mirror) {
        int n = resolveTbbNumaNode(numa_node, avoid_node);  // respect cpunodebind, avoid Stage-1
        auto c = tbb::task_arena::constraints{};
        if (n >= 0) c.set_numa_id(n);
        cpu_dfs_arena_ = new tbb::task_arena(c);
        cudaErrorCheck(cudaStreamCreateWithFlags(&cpu_dfs_stream_, cudaStreamNonBlocking));
        cudaErrorCheck(cudaStreamCreateWithFlags(&cpu_mask_stream_, cudaStreamNonBlocking));
        cudaErrorCheck(cudaEventCreate(&cpu_dfs_ev0_));
        cudaErrorCheck(cudaEventCreate(&cpu_dfs_ev1_));
        std::cout << "[CPU_DFS] NUMA-" << (n >= 0 ? std::to_string(n) : std::string("auto"))
                  << " arena ready (requested " << numa_node << ", avoid " << avoid_node
                  << "); initial ratio=" << cpu_dfs_ratio_ << ", min_rows=" << cpu_dfs_min_rows_ << "\n";
    }
}

// D2H flat_support_masks_ into the CPU mirror on the side stream so the CPU DFS
// can prune candidates with the support mask (mirrors the GPU's smask_read).
void GPUGraphLoader::SyncCPUMirrorMasks() {
    if (!cpu_mirror_ || !flat_support_masks_ || !cpu_mask_stream_) return;
    uint32_t dv_stride = SMASK_DV_STRIDE(DV_COUNT);
    cpu_mirror_->EnsureSmBuffer(dv_stride);
    // Copy only the [0, QE_COUNT*QV_COUNT) tiles the DFS actually reads (not all
    // MAX_VCOUNT). On its own stream so it does NOT block the frontier D2H.
    size_t words = (size_t)QE_COUNT * QV_COUNT * dv_stride;
    // tiles are strided by MAX_VCOUNT in the flat layout: copy per-ei runs.
    for (uint8_t e = 0; e < QE_COUNT; e++) {
        size_t off = (size_t)(e * MAX_VCOUNT) * dv_stride;
        cudaErrorCheck(cudaMemcpyAsync(cpu_mirror_->SmHostPtr() + off,
            flat_support_masks_ + off, (size_t)QV_COUNT * dv_stride * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, cpu_mask_stream_));
    }
}

// Split Step 5's final DFS between CPU ([0,k)) and GPU ([k,res_size_)), run them
// concurrently, join, and adapt the EMA ratio. Returns true if handled.
bool GPUGraphLoader::Step5CPUGPUSplit(const RelationsGPU& update_index,
                                      unsigned long long int& num_matches) {
    if (!cpu_mirror_ || !cpu_dfs_arena_) return false;
    if (res_size_ < cpu_dfs_min_rows_) return false;
    uint64_t k = (uint64_t)(cpu_dfs_ratio_ * (double)res_size_);
    if (k == 0 || k >= res_size_) return false;

    using Clock = std::chrono::high_resolution_clock;
    const uint8_t D = cur_depth_;
    const uint64_t cap = res_queue_.capability_;
    const uint64_t need = k * (uint64_t)D;

    // (1) ensure a PINNED host frontier buffer (true async D2H + faster CPU reads).
    //    Reserve to the next power of two so spiky k·res_size growth does NOT
    //    trigger a cudaFreeHost/cudaMallocHost (page-locking ~ms) every batch.
    if ((size_t)need > h_frontier_buf_cap_) {
        size_t cap = std::max((size_t)need, (size_t)8);
        cap = (size_t)std::exp2(std::ceil(std::log2((double)cap)));
        if (h_frontier_buf_) cudaErrorCheck(cudaFreeHost(h_frontier_buf_));
        cudaErrorCheck(cudaMallocHost((void**)&h_frontier_buf_, cap * sizeof(uint32_t)));
        h_frontier_buf_cap_ = cap;
    }
    // (2) D2H frontier rows [0,k) ASYNC on the side stream (overlaps the GPU launch)
    uint64_t start = res_ % cap;
    if (start + need <= cap) {
        cudaErrorCheck(cudaMemcpyAsync(h_frontier_buf_, res_queue_.array_ + start,
            need * sizeof(uint32_t), cudaMemcpyDeviceToHost, cpu_dfs_stream_));
    } else {
        uint64_t first = cap - start;
        cudaErrorCheck(cudaMemcpyAsync(h_frontier_buf_, res_queue_.array_ + start,
            first * sizeof(uint32_t), cudaMemcpyDeviceToHost, cpu_dfs_stream_));
        cudaErrorCheck(cudaMemcpyAsync(h_frontier_buf_ + first, res_queue_.array_,
            (need - first) * sizeof(uint32_t), cudaMemcpyDeviceToHost, cpu_dfs_stream_));
    }

    // (3) launch GPU share [k, res_size_) on the default stream — runs concurrently
    //     with the side-stream D2H (copy engine vs SM).
    cudaErrorCheck(cudaEventRecord(cpu_dfs_ev0_, 0));
    extendBFSAllBit<<<DIV_CEIL(res_size_ - k, NWARP_PER_BLOCK), BLOCK_DIM>>>(
        res_ + k * (uint64_t)D, res_size_ - k, new_res_, new_res_size_, 0,
        d_all_local_, update_index, d_edge_sm_ptrs_, D, QV_COUNT, false);
    cudaErrorCheck(cudaEventRecord(cpu_dfs_ev1_, 0));

    // (4) wait for D2H only (GPU keeps running), then CPU DFS concurrently with GPU.
    //     Also ensure the support-mask D2H (issued earlier on cpu_mask_stream_,
    //     overlapped with the BFS steps) is complete — returns immediately if done.
    cudaErrorCheck(cudaStreamSynchronize(cpu_dfs_stream_));
    if (cpu_mask_stream_) cudaErrorCheck(cudaStreamSynchronize(cpu_mask_stream_));
    auto tcpu0 = Clock::now();
    uint64_t cpu_cnt = cpuExtendBFSAllBitCountParallel(
        h_frontier_buf_, k, D, *cpu_mirror_, query_, plan_, *cpu_dfs_arena_);
    auto tcpu1 = Clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(tcpu1 - tcpu0).count();

    // (5) join GPU
    cudaErrorCheck(cudaEventSynchronize(cpu_dfs_ev1_));
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_,
        sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    float gpu_ms = 0.f;
    cudaErrorCheck(cudaEventElapsedTime(&gpu_ms, cpu_dfs_ev0_, cpu_dfs_ev1_));

    num_matches += h_new_res_size_ + cpu_cnt;

    // (6) EMA: balance CPU/GPU throughput => ratio = cpu_tp / (cpu_tp + gpu_tp)
    if (k > 0 && res_size_ - k > 0 && cpu_ms > 1e-6 && gpu_ms > 1e-6) {
        double cpu_tp = (double)k / cpu_ms;
        double gpu_tp = (double)(res_size_ - k) / gpu_ms;
        double rt = cpu_tp / (cpu_tp + gpu_tp);
        cpu_dfs_ratio_ = (float)(cpu_dfs_alpha_ * rt + (1.0 - cpu_dfs_alpha_) * cpu_dfs_ratio_);
        if (cpu_dfs_ratio_ < 0.05f) cpu_dfs_ratio_ = 0.05f;
        if (cpu_dfs_ratio_ > 0.95f) cpu_dfs_ratio_ = 0.95f;
    }
    return true;
}
#endif

void GPUGraphLoader::MatchingBitAll(
    const RelationsGPU& index_gpu,
    const RelationsGPU& update_index,
    const bool* edge_ok,
    unsigned long long int& num_matches) {

    using Clock = std::chrono::high_resolution_clock;
    double t_init = 0, t_step1 = 0, t_step2 = 0, t_step3 = 0, t_step4 = 0, t_step5 = 0;
    auto t_total_begin = Clock::now();

    /************************************ Init ************************************/
    auto t0 = Clock::now();
    if (!d_edge_ok_)
        cudaErrorCheck(cudaMalloc(&d_edge_ok_, sizeof(bool) * MAX_ECOUNT));
    cudaErrorCheck(cudaMemcpy(d_edge_ok_, edge_ok, sizeof(bool) * QE_COUNT, cudaMemcpyHostToDevice));
    // g_d_cf (per-data-vertex conflict-free) is uploaded separately by UploadConflictFree()
    SetupLocalIndexAll(index_gpu, update_index);
    res_queue_.Reset();
    bool oom = false;
    auto t1 = Clock::now();
    t_init = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 1: writeInitialResults ************************************/
    t0 = Clock::now();
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    new_depth_ = 2u;
    h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
    cudaErrorCheck(cudaDeviceSynchronize());

    dim3 init_grid(GRID_DIM, QE_COUNT);
    writeInitialResultsAllBit<<<init_grid, BLOCK_DIM>>>(
        d_all_local_, update_index, d_edge_sm_ptrs_,
        d_edge_ok_, new_res_, new_res_size_, h_max_new_res_size_
    );
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    res_queue_.Push(h_new_res_size_ * new_depth_);
    res_ = new_res_;
    res_size_ = h_new_res_size_;
    t1 = Clock::now();
    t_step1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (res_size_ == 0ul) return;
    cur_depth_ = new_depth_;

    /************************************ Step 2: BFS small-result loop ************************************/
    t0 = Clock::now();
    while (res_size_ < MIN_NRESULTS_TO_GPU && QV_COUNT - cur_depth_ > 1)
    {
        cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
        new_depth_ = cur_depth_ + 1;
        new_res_ = res_queue_.TryMax();
        h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
        cudaErrorCheck(cudaDeviceSynchronize());

        extendBFSAllBit<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
            res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_,
            d_all_local_, update_index, d_edge_sm_ptrs_, cur_depth_, new_depth_, true
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        if (h_new_res_size_ >= h_max_new_res_size_)
        {
            oom = true;
            break;
        }
        else
        {
            res_queue_.Push(h_new_res_size_ * new_depth_);
            res_queue_.Pop(res_size_ * cur_depth_);
            res_ = new_res_;
            res_size_ = h_new_res_size_;
            if (res_size_ == 0ul) return;
            cur_depth_ = new_depth_;
        }
    }
    t1 = Clock::now();
    t_step2 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 3: multi-level BFS ************************************/
    t0 = Clock::now();
    if (!oom && plan_.global_cartesian_product_info_[cur_depth_] != Plan::CartesianProductType::TreeCartesianProduct && QV_COUNT - cur_depth_ > 2)
    {
        new_depth_ = cur_depth_;
        while (
            new_depth_ < QV_COUNT &&
            plan_.global_cartesian_product_info_[new_depth_] != Plan::CartesianProductType::TreeCartesianProduct &&
            plan_.global_cartesian_product_info_[new_depth_] != Plan::CartesianProductType::TreeSingle
        ) {
            new_depth_++;
        }
        if (QV_COUNT - new_depth_ > 0)
        {
            cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
            new_res_ = res_queue_.TryMax();
            h_max_new_res_size_ = res_queue_.GetFree() / new_depth_;
            cudaErrorCheck(cudaDeviceSynchronize());

            extendBFSAllBit<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
                res_, res_size_, new_res_, new_res_size_, h_max_new_res_size_,
                d_all_local_, update_index, d_edge_sm_ptrs_, cur_depth_, new_depth_, true
            );
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            if (h_new_res_size_ >= h_max_new_res_size_)
            {
                oom = true;
            }
            else
            {
                res_queue_.Push(h_new_res_size_ * new_depth_);
                res_queue_.Pop(res_size_ * cur_depth_);
                res_ = new_res_;
                res_size_ = h_new_res_size_;
                if (res_size_ == 0ul) return;
                cur_depth_ = new_depth_;
            }
        }
    }
    t1 = Clock::now();
    t_step3 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 4: Cartesian product analysis ************************************/
    t0 = Clock::now();
    bool enumerate_cartesian_product = plan_.global_cartesian_product_info_[cur_depth_] == Plan::CartesianProductType::TreeCartesianProduct && res_size_ < SIZE_SPACE - 1;
    if (enumerate_cartesian_product)
    {
        GetNumTreeAllBit<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_,
            d_all_local_, update_index, d_edge_sm_ptrs_,
            QV_COUNT - cur_depth_
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        CUB(cub::DeviceReduce::Max(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, max_res_size_cartesian_product_, res_size_));
        cudaErrorCheck(cudaDeviceSynchronize());
        CUB(cub::DeviceScan::ExclusiveSum(d_temp_storage_, temp_storage_bytes_,
            res_size_cartesian_product_, res_size_cartesian_product_, res_size_ + 1));
        cudaErrorCheck(cudaDeviceSynchronize());
        unsigned long h_max, h_total;
        cudaErrorCheck(cudaMemcpy(&h_max, max_res_size_cartesian_product_, sizeof(unsigned long), cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaMemcpy(&h_total, res_size_cartesian_product_ + res_size_, sizeof(unsigned long), cudaMemcpyDeviceToHost));

        const float avg_res_size = (float)h_total / res_size_;
        const float ratio = (avg_res_size > 0) ? (float)h_max / avg_res_size : 0;

        if (ratio < 50.f && avg_res_size >= MIN_NRESULTS_TO_GPU)
        {
            enumerate_cartesian_product = false;
        }
    }
    t1 = Clock::now();
    t_step4 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    /************************************ Step 5: final enumeration ************************************/
    t0 = Clock::now();
    cudaErrorCheck(cudaMemset(new_res_size_, 0u, sizeof(unsigned long long int)));
    new_res_ = res_queue_.TryMax();
    cudaErrorCheck(cudaDeviceSynchronize());

#ifdef ENABLE_CPU_DFS
    last_step5_cartesian_ = enumerate_cartesian_product;
#endif
    if (enumerate_cartesian_product)
    {
        // cout << "enumerateCartesianProductAllBit" << endl;
        enumerateCartesianProductAllBit<<<GRID_DIM, BLOCK_DIM>>>(
            res_, res_size_, res_size_cartesian_product_, res_size_cartesian_product_ + res_size_,
            d_all_local_, update_index, d_edge_sm_ptrs_,
            new_res_size_, QV_COUNT - cur_depth_
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        num_matches += h_new_res_size_;
    }
    else
    {
        bool split_handled = false;
#ifdef ENABLE_CPU_DFS
        split_handled = Step5CPUGPUSplit(update_index, num_matches);
#endif
        if (!split_handled)
        {
#ifdef USE_GLOBAL_RQ
            cout << "extendBFSAllBitGlobal" << endl;
            unsigned long long grid_size = min(
                DIV_CEIL(res_size_, (unsigned long long)NWARP_PER_BLOCK),
                (unsigned long long)GRID_DIM);
            extendBFSAllBitGlobal<<<grid_size, BLOCK_DIM>>>(
                res_, res_size_, new_res_, new_res_size_, 0,
                d_all_local_, update_index, d_edge_sm_ptrs_, cur_depth_, QV_COUNT, false,
                d_global_rq_
            );
#else
            // cout << "extendBFSAllBitBalance" << endl;
            extendBFSAllBit<<<DIV_CEIL(res_size_, NWARP_PER_BLOCK), BLOCK_DIM>>>(
                res_, res_size_, new_res_, new_res_size_, 0,
                d_all_local_, update_index, d_edge_sm_ptrs_, cur_depth_, QV_COUNT, false
            );
#endif
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(&h_new_res_size_, new_res_size_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
            num_matches += h_new_res_size_;
        }
    }
    t1 = Clock::now();
    t_step5 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t_total_end = Clock::now();
    double t_total = std::chrono::duration<double, std::milli>(t_total_end - t_total_begin).count();
    // cout << "[MatchingBitAll Timing] "
    //      << "init=" << t_init << "ms, "
    //      << "step1(writeInitial)=" << t_step1 << "ms, "
    //      << "step2(BFS_small)=" << t_step2 << "ms, "
    //      << "step3(BFS_multi)=" << t_step3 << "ms, "
    //      << "step4(cartesian_analysis)=" << t_step4 << "ms, "
    //      << "step5(enumeration)=" << t_step5 << "ms, "
    //      << "total=" << t_total << "ms" << endl;
}
#endif
