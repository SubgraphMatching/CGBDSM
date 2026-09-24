#include "calig_helper_gamma.h"
#include "utils/globals.h"
#include "utils/cuda_helpers.h"
#include "utils/constants.h"
#include "kernels/indexing.h"
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_arena.h>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <thread>
using namespace std;
#define cErr(errcode) { gpuAssert((errcode), __FILE__, __LINE__); }

CaLiGHelperGamma::CaLiGHelperGamma(CaLiG *calig, RelationsGPU* gpu_index,
                                    RelationsGPU* gpu_update_index,
                                    RelationsGPU* gpu_local_update_index,
                                    Plan *plan, uint32_t num_batches,
                                    MemPool<uint32_t>* nbr_mem_pool) {
    for (uint32_t u_q = 0; u_q < calig->Q.size(); ++u_q)
        for (uint32_t v_q : calig->Q[u_q].nei)
            num_edges++;
    this->calig = calig;
    this->gpu_index = gpu_index;
    this->gpu_update_index = gpu_update_index;
    this->gpu_local_update_index = gpu_local_update_index;
    this->plan = plan;
    this->nbr_mem_pool_ = nbr_mem_pool;
    this->tmp_el.resize(calig->getNumThreads());
    this->global_vid2newidx.clear();
    this->dense_vid_map_.resize(calig->G.size(), UINT32_MAX);
    this->gpu_capabilities_.resize(num_edges * 12, 0);
    el_host_stage_.resize(num_batches);
    del_host_stage_.resize(num_batches);
    el_update_host_stage_.resize(num_batches);
    G_UPDATE_FLAT_.resize(num_batches);
    vertex_mapping_changed_.resize(num_batches, false);
    num_valid_vertices_buf_.resize(num_batches, 0);
    for (uint32_t b = 0; b < num_batches; b++)
        G_UPDATE_FLAT_[b].resize(num_edges);
    edge_ok_per_batch_.resize(num_batches);
#ifdef ENABLE_CPU_DFS
    cpu_mirror_ = new CPUIndexMirror(plan->query_, *plan, num_edges);
#endif
}

template <typename T>
void reallocateCudaPtr(T* &ptr, size_t old_count, size_t new_count, size_t &capacity) {
    if (new_count <= capacity) {
        if (new_count > old_count)
            cErr(cudaMemset(ptr + old_count, 0, (new_count - old_count) * sizeof(T)));
        return;
    }

    size_t new_capacity = max((size_t)(new_count * 1.5), (size_t)exp2(ceil(log2(new_count))));
    new_capacity = max(new_capacity, (size_t)8ul);

    T* new_ptr = nullptr;
    cErr(cudaMalloc(&new_ptr, new_capacity * sizeof(T)));
    cErr(cudaMemset(new_ptr, 0, new_capacity * sizeof(T)));

    if (ptr != nullptr && old_count > 0) {
        size_t copy_count = min(old_count, capacity);
        cErr(cudaMemcpy(new_ptr, ptr, copy_count * sizeof(T), cudaMemcpyDeviceToDevice));
        cErr(cudaFree(ptr));
    }

    ptr = new_ptr;
    capacity = new_capacity;
}


void CaLiGHelperGamma::UpdateVertexMappingCPU(uint32_t batch) {
    uint32_t num_data_vertices = calig->G.size();
    int max_threads = calig->getNumThreads();
    vector<vector<uint32_t>> local_new_vertices(max_threads);

    tbb::parallel_for(0u, num_data_vertices, [&](uint32_t u_data) {
        if (calig->G_Li[u_data] == 0) return;
        if (global_vid2newidx.find(u_data) != global_vid2newidx.end()) return;
        local_new_vertices[tbb::this_task_arena::current_thread_index()].push_back(u_data);
    });

    ska::flat_hash_set<uint32_t> new_vertices_set;
    for (int tid = 0; tid < max_threads; ++tid)
        for (uint32_t v : local_new_vertices[tid])
            new_vertices_set.insert(v);
    for (uint32_t v : new_vertices_set) {
        uint32_t new_idx = global_vid2newidx.size();
        global_vid2newidx[v] = new_idx;
        dense_vid_map_[v] = new_idx;
        valid_vid_pairs_.emplace_back(v, new_idx);
    }

    if (num_valid_vertices_buf_[batch] == global_vid2newidx.size()) {
        vertex_mapping_changed_[batch] = false;
        return;
    }
    num_valid_vertices_buf_[batch] = global_vid2newidx.size();
    vertex_mapping_changed_[batch] = true;
}

void CaLiGHelperGamma::AllocateFromMappingGPU(uint32_t batch) {
    if (!vertex_mapping_changed_[batch]) return;

    for (int gamma_eidx = 0; gamma_eidx < (int)num_edges; gamma_eidx++) {
        reallocateCudaPtr(gpu_index->sizes_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[gamma_eidx * 3]);
        reallocateCudaPtr(gpu_index->nbrs_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[gamma_eidx * 3 + 1]);
        reallocateCudaPtr(gpu_index->capability_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[gamma_eidx * 3 + 2]);

        reallocateCudaPtr(gpu_update_index->sizes_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[num_edges * 4 + gamma_eidx * 2]);
        reallocateCudaPtr(gpu_update_index->nbrs_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[num_edges * 4 + gamma_eidx * 2 + 1]);

        reallocateCudaPtr(gpu_local_update_index->sizes_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[num_edges * 8 + gamma_eidx * 3]);
        reallocateCudaPtr(gpu_local_update_index->nbrs_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[num_edges * 8 + gamma_eidx * 3 + 1]);
        reallocateCudaPtr(gpu_local_update_index->capability_[gamma_eidx], num_valid_vertices_buf_[batch] + 1, num_valid_vertices_buf_[batch] + 1, gpu_capabilities_[num_edges * 8 + gamma_eidx * 3 + 2]);
    }

    uint32_t nv = num_valid_vertices_buf_[batch];
    DV_COUNT = nv;
    cudaErrorCheck(cudaMemcpyToSymbol(C_DV_COUNT, &nv, sizeof(uint32_t)));
#ifdef ENABLE_CPU_DFS
    if (cpu_mirror_) cpu_mirror_->ResizeForDVCount(nv);
#endif
    vertex_mapping_changed_[batch] = false;
}

void CaLiGHelperGamma::buildGlobalVertexMapping() {
    uint32_t num_data_vertices = calig->G.size();
    int max_threads = calig->getNumThreads();
    vector<vector<uint32_t>> local_new_vertices(max_threads);

    tbb::parallel_for(0u, num_data_vertices, [&](uint32_t u_data) {
        if (calig->G_Li[u_data] == 0) return;
        if (global_vid2newidx.find(u_data) != global_vid2newidx.end()) return;
        local_new_vertices[tbb::this_task_arena::current_thread_index()].push_back(u_data);
    });

    ska::flat_hash_set<uint32_t> new_vertices_set;
    for (int tid = 0; tid < max_threads; ++tid)
        for (uint32_t v : local_new_vertices[tid])
            new_vertices_set.insert(v);
    for (uint32_t v : new_vertices_set) {
        uint32_t new_idx = global_vid2newidx.size();
        global_vid2newidx[v] = new_idx;
        // if(new_idx == 5344 || new_idx == 1093 || new_idx == 25693)
        //     printf("%d:%d\n", v, new_idx);
        dense_vid_map_[v] = new_idx;
        valid_vid_pairs_.emplace_back(v, new_idx);
    }

    if (num_valid_vertices_buf_[0] == global_vid2newidx.size()) {
        return;
    }

    size_t old_num_valid_vertices = num_valid_vertices_buf_[0];
    num_valid_vertices_buf_[0] = global_vid2newidx.size();

    for (int gamma_eidx = 0; gamma_eidx < (int)num_edges; gamma_eidx++) {
        reallocateCudaPtr(gpu_index->sizes_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[gamma_eidx * 3]);
        reallocateCudaPtr(gpu_index->nbrs_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[gamma_eidx * 3 + 1]);
        reallocateCudaPtr(gpu_index->capability_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[gamma_eidx * 3 + 2]);

        reallocateCudaPtr(gpu_update_index->sizes_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[num_edges * 4 + gamma_eidx * 2]);
        reallocateCudaPtr(gpu_update_index->nbrs_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[num_edges * 4 + gamma_eidx * 2 + 1]);

        reallocateCudaPtr(gpu_local_update_index->sizes_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[num_edges * 8 + gamma_eidx * 3]);
        reallocateCudaPtr(gpu_local_update_index->nbrs_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[num_edges * 8 + gamma_eidx * 3 + 1]);
        reallocateCudaPtr(gpu_local_update_index->capability_[gamma_eidx], old_num_valid_vertices + 1, num_valid_vertices_buf_[0] + 1, gpu_capabilities_[num_edges * 8 + gamma_eidx * 3 + 2]);
    }

    DV_COUNT = num_valid_vertices_buf_[0];
    cudaErrorCheck(cudaMemcpyToSymbol(C_DV_COUNT, &num_valid_vertices_buf_[0], sizeof(uint32_t)));
#ifdef ENABLE_CPU_DFS
    if (cpu_mirror_) cpu_mirror_->ResizeForDVCount(num_valid_vertices_buf_[0]);
#endif
}

uint32_t CaLiGHelperGamma::getNewVertexId(uint32_t old_vid) {
    if (old_vid < dense_vid_map_.size()) return dense_vid_map_[old_vid];
    return UINT32_MAX;
}

// std::vector<uint8_t> CaLiGHelperGamma::BuildCompressedConflictFree2D() {
//     // Convert per-(pattern-edge, update-edge) conflicts to the per-(ei, compressed
//     // data vertex) GPU structure: a vertex is "cannot-skip" (0) on ei iff it is an
//     // endpoint of some update edge k that conflicts on ei. Default is conflict-free (1).
//     const auto& ec  = calig->getEdgeConflict();
//     const uint32_t np = calig->getDCNumPairs();
//     const uint32_t st = calig->getDCStart();
//     const auto& upd = calig->getUpdate();
//     const uint32_t comp_dv = static_cast<uint32_t>(global_vid2newidx.size());
//     const uint32_t qe = plan->query_.ecount_;
//     std::vector<uint8_t> cf2d(static_cast<size_t>(qe) * comp_dv, 1);
//     for (uint32_t k = 0; k < np; ++k) {
//         const uint32_t v1 = static_cast<uint32_t>(upd[st + 2 * k]);
//         const uint32_t v2 = static_cast<uint32_t>(upd[st + 2 * k + 1]);
//         auto it1 = global_vid2newidx.find(v1);
//         auto it2 = global_vid2newidx.find(v2);
//         const bool has1 = it1 != global_vid2newidx.end();
//         const bool has2 = it2 != global_vid2newidx.end();
//         if (!has1 && !has2) continue;
//         for (uint32_t ei = 0; ei < qe; ++ei) {
//             if (!ec[static_cast<size_t>(ei) * np + k]) continue;
//             if (has1) cf2d[static_cast<size_t>(ei) * comp_dv + it1->second] = 0;
//             if (has2) cf2d[static_cast<size_t>(ei) * comp_dv + it2->second] = 0;
//         }
//     }
//     return cf2d;
// }

CaLiGHelperGamma::~CaLiGHelperGamma() {
#ifdef ENABLE_CPU_DFS
    delete cpu_mirror_;
    cpu_mirror_ = nullptr;
#endif
}

__global__ void generateSizePtr(uint64_t data[], uint32_t data_32[], uint32_t **sizes, uint32_t ***nbrs, uint32_t size) {
    uint32_t global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t block_offset = gridDim.x * blockDim.x;
    for (uint32_t i = global_thread_id; i < size; i += block_offset) {
        uint64_t val = data[i];
        uint32_t gamma_eidx = (uint32_t)(val >> 59);
        uint32_t u = (uint32_t)((val >> 32) & 0x07FFFFFF);
        uint32_t v = (uint32_t)val;
        data_32[i] = v;
        atomicAdd(&sizes[gamma_eidx][u], 1);
        if (i == 0 || ((data[i - 1] >> 32) != (data[i] >> 32)))
            nbrs[gamma_eidx][u] = &data_32[i];
    }
}

void CaLiGHelperGamma::ConvertGlobalIndex(uint32_t *cardinalities, float *degrees) {
    buildGlobalVertexMapping();
    uint32_t max_threads = calig->getNumThreads();
    std::vector<uint32_t> cardinalities_tmp(num_edges * max_threads, 0);
    std::vector<uint32_t> cardinalities_nodes_tmp(num_edges * max_threads, 0);
    std::vector<uint32_t> cardinalities_nodes_tmp_sum(num_edges, 0);
    uint32_t num_data_vertices = calig->G.size(), num_qv = calig->Q.size();
    el_dev.clear(); el_32_dev.clear();
    for(uint32_t i = 0; i < max_threads; i++) tmp_el[i].clear();

    tbb::parallel_for(0u, (uint32_t)valid_vid_pairs_.size(), [&](uint32_t idx) {
        int tid = tbb::this_task_arena::current_thread_index();
        uint32_t u_data = valid_vid_pairs_[idx].first;
        uint32_t new_u = valid_vid_pairs_[idx].second;
        if (calig->G_Li[u_data] == 0) return;

        for (auto const& [u_q, neighbors_map] : calig->G[u_data].cand) {
            if (!(calig->G_Li[u_data] & (1 << u_q))) continue;
            for (auto const& [v_q, current_u_neighbors] : neighbors_map) {
                uint32_t gamma_eidx = plan->query_.eidx_[u_q * plan->query_.vcount_ + v_q];
                uint32_t tmp_size = 0;
                for (auto v_data : current_u_neighbors) {
                    uint32_t new_v = getNewVertexId(v_data);
                    if (new_v == UINT32_MAX) continue;
                    cardinalities_tmp[gamma_eidx * max_threads + tid] ++;
                    tmp_size ++;
                    uint64_t data = (uint64_t)((uint64_t) gamma_eidx << 59 | (uint64_t) new_u << 32 | new_v);
                    tmp_el[tid].push_back(data);
                }
                cardinalities_nodes_tmp[gamma_eidx * max_threads + tid] += (tmp_size != 0);
            }
        }
    });
    for(uint32_t tid = 0; tid < max_threads; tid++) {
        for(uint32_t gamma_eidx = 0; gamma_eidx < num_edges; gamma_eidx++) {
            cardinalities[gamma_eidx] += cardinalities_tmp[gamma_eidx * max_threads + tid];
            cardinalities_nodes_tmp_sum[gamma_eidx] += cardinalities_nodes_tmp[gamma_eidx * max_threads + tid];
        }
        el_dev.insert(el_dev.end(), tmp_el[tid].begin(), tmp_el[tid].end());
    }

#ifdef ENABLE_CPU_DFS
    // Build the host mirror from the same host-encoded edges (tmp_el) that feed
    // el_dev — no D2H. Sorted on host; identical content to the GPU's el_dev.
    if (cpu_mirror_) {
        auto _t0 = std::chrono::high_resolution_clock::now();
        // Parallel concatenation of per-thread tmp_el into one flat vector.
        std::vector<size_t> offs(max_threads + 1, 0);
        for (uint32_t tid = 0; tid < max_threads; tid++)
            offs[tid + 1] = offs[tid] + tmp_el[tid].size();
        std::vector<uint64_t> host_edges(offs[max_threads]);
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, max_threads, 4),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t tid = r.begin(); tid < r.end(); tid++) {
                    auto& hv = tmp_el[tid];
                    std::copy(hv.data(), hv.data() + hv.size(), host_edges.data() + offs[tid]);
                }
            });
        cpu_mirror_->ResizeForDVCount(num_valid_vertices_buf_[0]);
        tbb::parallel_sort(host_edges.begin(), host_edges.end());
        cpu_mirror_->InitFromSortedEdges(host_edges);
        auto _t1 = std::chrono::high_resolution_clock::now();
        cpu_mirror_build_ms_ += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    }
#endif

    thrust::sort(el_dev.begin(), el_dev.end());
    el_32_dev.resize(el_dev.size());
    for(uint32_t gamma_eidx = 0; gamma_eidx < num_edges; gamma_eidx++) {
        cudaMemset(gpu_index->sizes_[gamma_eidx], 0, (num_valid_vertices_buf_[0] + 1) * sizeof(uint32_t));
        cudaMemset(gpu_index->nbrs_[gamma_eidx], 0, (num_valid_vertices_buf_[0] + 1) * sizeof(uint32_t*));
        degrees[gamma_eidx] = (cardinalities_nodes_tmp_sum[gamma_eidx] > 0) ? (float)cardinalities[gamma_eidx] / cardinalities_nodes_tmp_sum[gamma_eidx] : 0.0f;
    }

    generateSizePtr<<<min((uint64_t)96 * 8, (el_dev.size() + 31) / 32), 32>>>(
        thrust::raw_pointer_cast(el_dev.data()),
        thrust::raw_pointer_cast(el_32_dev.data()),
        gpu_index->sizes_,
        gpu_index->nbrs_,
        el_dev.size());
    cErr(cudaDeviceSynchronize());

    uint32_t nv = num_valid_vertices_buf_[0];
    for (uint32_t gamma_eidx = 0; gamma_eidx < num_edges; gamma_eidx++) {
        thrust::device_vector<uint32_t*> old_nbrs(nv + 1);
        cudaMemcpy(thrust::raw_pointer_cast(old_nbrs.data()),
                   gpu_index->nbrs_[gamma_eidx],
                   (nv + 1) * sizeof(uint32_t*), cudaMemcpyDeviceToDevice);

        cudaMemset(gpu_index->capability_[gamma_eidx], 0, (nv + 1) * sizeof(uint32_t));
        setCapabilitiesFromSizes<<<GRID_DIM, BLOCK_DIM>>>(
            gpu_index->sizes_[gamma_eidx], gpu_index->capability_[gamma_eidx], nv + 1);
        cErr(cudaDeviceSynchronize());
        roundCapabilities<<<GRID_DIM, BLOCK_DIM>>>(
            gpu_index->capability_[gamma_eidx], nv + 1);
        cErr(cudaDeviceSynchronize());

        allocateFromMemPool<<<GRID_DIM, BLOCK_DIM>>>(
            *gpu_index, gamma_eidx, nv + 1, *nbr_mem_pool_);
        cErr(cudaDeviceSynchronize());

        if (nbr_mem_pool_->OutOfMemory()) {
            fprintf(stderr, "[ConvertGlobalIndex] MemPool OOM during allocation!\n");
            exit(-1);
        }

        copyFromFlatCSR<<<GRID_DIM, BLOCK_DIM>>>(
            thrust::raw_pointer_cast(old_nbrs.data()),
            gpu_index->sizes_[gamma_eidx],
            gpu_index->nbrs_[gamma_eidx], nv + 1);
        cErr(cudaDeviceSynchronize());
    }

    if (nbr_mem_pool_->OutOfMemory()) {
        fprintf(stderr, "[ConvertGlobalIndex] MemPool OOM!\n");
        exit(-1);
    }

    el_dev.clear();
    el_32_dev.clear();
    el_dev.shrink_to_fit();
    el_32_dev.shrink_to_fit();
}

// ============================================================================
void CaLiGHelperGamma::PrepareUpdateAll(uint32_t batch) {
    UpdateVertexMappingCPU(batch);

    const auto& ins = calig->getInsDeltas(batch);
    const auto& del = calig->getDelDeltas(batch);
    uint32_t max_threads = calig->getNumThreads();

    for(uint32_t i = 0; i < max_threads; i++) tmp_el[i].clear();
    el_host_stage_[batch].clear();

    tbb::parallel_for(0u, (uint32_t)ins.size(), [&](uint32_t i) {
        int tid = tbb::this_task_arena::current_thread_index();
        auto& d = ins[i];
        if (!(calig->G_Li[d.vi] & (1u << d.ui))) return;
        uint32_t gamma_eidx = plan->query_.eidx_[d.ui * plan->query_.vcount_ + d.uj];
        uint32_t new_u = getNewVertexId(d.vi);
        if (new_u == UINT32_MAX) return;
        uint32_t new_v = getNewVertexId(d.vj);
        if (new_v == UINT32_MAX) return;
        tmp_el[tid].push_back((uint64_t)((uint64_t)gamma_eidx << 59 | (uint64_t)new_u << 32 | new_v));
    });

    size_t total_size = 0;
    for(uint32_t tid = 0; tid < max_threads; tid++) total_size += tmp_el[tid].size();
    el_host_stage_[batch].reserve(total_size);
    for(uint32_t tid = 0; tid < max_threads; tid++)
        el_host_stage_[batch].insert(el_host_stage_[batch].end(), tmp_el[tid].begin(), tmp_el[tid].end());

    del_host_stage_[batch].clear();
    del_host_stage_[batch].reserve(del.size());
    for (auto& d : del) {
        uint32_t gamma_eidx = plan->query_.eidx_[d.ui * plan->query_.vcount_ + d.uj];
        uint32_t new_u = getNewVertexId(d.vi);
        if (new_u == UINT32_MAX) continue;
        uint32_t new_v = getNewVertexId(d.vj);
        if (new_v == UINT32_MAX) continue;
        del_host_stage_[batch].push_back((uint64_t)((uint64_t)gamma_eidx << 59 | (uint64_t)new_u << 32 | new_v));
    }
}

void CaLiGHelperGamma::PrepareUpdateIndex(uint32_t batch) {
    const auto& updates = calig->getUpdateDeltas(batch);
    uint32_t max_threads = calig->getNumThreads();

    for(uint32_t i = 0; i < max_threads; i++) tmp_el[i].clear();
    el_update_host_stage_[batch].clear();

    for (uint32_t e = 0; e < num_edges; e++)
        G_UPDATE_FLAT_[batch][e].clear();

    struct EdgeItem { uint32_t qe_idx; uint32_t u_data; uint32_t v_data; };
    std::vector<std::vector<EdgeItem>> local_flat(max_threads);

    tbb::parallel_for(0u, (uint32_t)updates.size(), [&](uint32_t i) {
        int tid = tbb::this_task_arena::current_thread_index();
        auto& d = updates[i];
        uint32_t gamma_eidx = plan->query_.eidx_[d.ui * plan->query_.vcount_ + d.uj];
        uint32_t new_u = getNewVertexId(d.vi);
        if (new_u == UINT32_MAX) return;
        uint32_t new_v = getNewVertexId(d.vj);
        if (new_v == UINT32_MAX) return;
        uint64_t data = (uint64_t)((uint64_t)gamma_eidx << 59 | (uint64_t)new_u << 32 | new_v);
        tmp_el[tid].push_back(data);
        local_flat[tid].push_back(EdgeItem{gamma_eidx, new_u, new_v});
    });

    size_t total_size = 0;
    for(uint32_t tid = 0; tid < max_threads; tid++) total_size += tmp_el[tid].size();
    el_update_host_stage_[batch].reserve(total_size);
    for(uint32_t tid = 0; tid < (uint32_t)max_threads; tid++) {
        el_update_host_stage_[batch].insert(el_update_host_stage_[batch].end(), tmp_el[tid].begin(), tmp_el[tid].end());
        for (auto &item : local_flat[tid])
            G_UPDATE_FLAT_[batch][item.qe_idx].push_back({item.u_data, item.v_data});
    }

    // Compute per-edge emptiness flags from CPU data (replaces GPU CUB reduction)
    for (uint8_t e = 0; e < QE_COUNT; e++) {
        uint8_t dir1 = plan->query_.qe_eidx_[e].first;
        uint8_t dir2 = plan->query_.qe_eidx_[e].second;
        edge_ok_per_batch_[batch][e] = !G_UPDATE_FLAT_[batch][dir1].empty()
                                     && !G_UPDATE_FLAT_[batch][dir2].empty();
    }
}
void CaLiGHelperGamma::mergeSortedEdgesToIndex(
    thrust::device_vector<uint64_t>& sorted_edges,
    bool is_insertion,
    uint32_t batch
) {
    if (sorted_edges.empty()) return;
    uint32_t nv = num_valid_vertices_buf_[batch];

    tmp_merge_32_.resize(sorted_edges.size());
    for (uint32_t eidx = 0; eidx < num_edges; eidx++) {
        cudaMemset(gpu_update_index->sizes_[eidx], 0, (nv + 1) * sizeof(uint32_t));
        cudaMemset(gpu_update_index->nbrs_[eidx], 0, (nv + 1) * sizeof(uint32_t*));
    }
    generateSizePtr<<<min((uint64_t)96 * 8, (sorted_edges.size() + 31) / 32), 32>>>(
        thrust::raw_pointer_cast(sorted_edges.data()),
        thrust::raw_pointer_cast(tmp_merge_32_.data()),
        gpu_update_index->sizes_,
        gpu_update_index->nbrs_,
        (uint32_t)sorted_edges.size());
    cErr(cudaDeviceSynchronize());

    for (uint32_t eidx = 0; eidx < num_edges; eidx++) {
        if (is_insertion)
            mergeCSROntoGraph<<<GRID_DIM, BLOCK_DIM>>>(
                *gpu_update_index, *gpu_index, eidx, *nbr_mem_pool_, nv + 1);
        else
            removeCSROromGraph<<<GRID_DIM, BLOCK_DIM>>>(
                *gpu_update_index, *gpu_index, eidx, nv + 1);
    }

    if (is_insertion) {
        cErr(cudaDeviceSynchronize());
        if (nbr_mem_pool_->OutOfMemory()) {
            fprintf(stderr, "[mergeSortedEdgesToIndex] MemPool OOM!\n");
            exit(-1);
        }
    }
}

void CaLiGHelperGamma::TransferUpdateAll(uint32_t batch) {
    AllocateFromMappingGPU(batch);

    auto& ins_host = el_host_stage_[batch];
    if (!ins_host.empty()) {
        tmp_sort_dev_.assign(ins_host.begin(), ins_host.end());   // H2D
        thrust::sort(tmp_sort_dev_.begin(), tmp_sort_dev_.end()); // GPU sort
        auto new_end = thrust::unique(tmp_sort_dev_.begin(), tmp_sort_dev_.end()); // dedup
        tmp_sort_dev_.erase(new_end, tmp_sort_dev_.end());
        mergeSortedEdgesToIndex(tmp_sort_dev_, true, batch);       // mergeCSROntoGraph
    }

    auto& del_host = del_host_stage_[batch];
    if (!del_host.empty()) {
        tmp_sort_dev_.assign(del_host.begin(), del_host.end());   // H2D
        thrust::sort(tmp_sort_dev_.begin(), tmp_sort_dev_.end()); // GPU sort
        mergeSortedEdgesToIndex(tmp_sort_dev_, false, batch);     // removeCSROromGraph
    }
#ifdef ENABLE_CPU_DFS
    // Mirror the same ins/del deltas into the host index (sequential; routes
    // through the NUMA-bound arena via CPUIndexMirror::run()). The per-call
    // std::thread pipeline was net-negative on orkut (spawn overhead dominated).
    if (cpu_mirror_) {
        auto _t0 = std::chrono::high_resolution_clock::now();
        cpu_mirror_->MergeInsert(el_host_stage_[batch]);
        auto _t1 = std::chrono::high_resolution_clock::now();
        cpu_mirror_ins_ms_ += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        cpu_mirror_->MergeRemove(del_host_stage_[batch]);
        auto _t2 = std::chrono::high_resolution_clock::now();
        cpu_mirror_del_ms_ += std::chrono::duration<double, std::milli>(_t2 - _t1).count();
    }
#endif
}

void CaLiGHelperGamma::UpdateGlobalIndex(uint32_t batch, uint8_t cur_i) {
    const uint32_t qe_idx1 = plan->query_.qe_eidx_[cur_i].first;
    const uint32_t qe_idx2 = plan->query_.qe_eidx_[cur_i].second;

    // Encode 2 directions' compressed ID pairs as 64-bit values
    update_host_enc_.clear();
    for (uint32_t qe_idx : {qe_idx1, qe_idx2}) {
        for (const auto& [new_u, new_v] : G_UPDATE_FLAT_[batch][qe_idx]) {
            update_host_enc_.push_back((uint64_t)((uint64_t)qe_idx << 59 | (uint64_t)new_u << 32 | new_v));
        }
    }
    if (update_host_enc_.empty()) return;

    tmp_sort_dev_ = update_host_enc_;                              // H2D
    thrust::sort(tmp_sort_dev_.begin(), tmp_sort_dev_.end());      // GPU sort
    auto ue = thrust::unique(tmp_sort_dev_.begin(), tmp_sort_dev_.end());
    tmp_sort_dev_.erase(ue, tmp_sort_dev_.end());
    mergeSortedEdgesToIndex(tmp_sort_dev_, true, batch);            // mergeCSROntoGraph
}

void CaLiGHelperGamma::BatchUpdateGlobalIndex(uint32_t batch) {
    update_host_enc_.clear();
    // Collect update deltas for all query edge directions
    for (uint8_t e = 0; e < QE_COUNT; e++) {
        uint32_t qe_idx1 = plan->query_.qe_eidx_[e].first;
        uint32_t qe_idx2 = plan->query_.qe_eidx_[e].second;
        for (uint32_t qe_idx : {qe_idx1, qe_idx2}) {
            for (const auto& [new_u, new_v] : G_UPDATE_FLAT_[batch][qe_idx]) {
                update_host_enc_.push_back((uint64_t)((uint64_t)qe_idx << 59 | (uint64_t)new_u << 32 | new_v));
            }
        }
    }
    if (update_host_enc_.empty()) return;

    tmp_sort_dev_ = update_host_enc_;                              // H2D
    thrust::sort(tmp_sort_dev_.begin(), tmp_sort_dev_.end());      // GPU sort
    auto new_end = thrust::unique(tmp_sort_dev_.begin(), tmp_sort_dev_.end()); // dedup
    tmp_sort_dev_.erase(new_end, tmp_sort_dev_.end());
    mergeSortedEdgesToIndex(tmp_sort_dev_, true, batch);            // mergeCSROntoGraph
#ifdef ENABLE_CPU_DFS
    if (cpu_mirror_) {
        auto _t0 = std::chrono::high_resolution_clock::now();
        cpu_mirror_->MergeInsert(std::vector<uint64_t>(
            update_host_enc_.data(), update_host_enc_.data() + update_host_enc_.size()));
        auto _t1 = std::chrono::high_resolution_clock::now();
        cpu_mirror_batch_ms_ += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    }
#endif
}

bool CaLiGHelperGamma::TransferUpdateIndex(uint32_t batch) {
    el_update_dev = el_update_host_stage_[batch];
    if(el_update_dev.size() == 0)
        return false;
    thrust::sort(el_update_dev.begin(), el_update_dev.end());
    el_update_32_dev.resize(el_update_dev.size());

    for(uint32_t gamma_eidx = 0; gamma_eidx < num_edges; gamma_eidx++) {
        cudaMemset(gpu_update_index->sizes_[gamma_eidx], 0, (num_valid_vertices_buf_[batch] + 1) * sizeof(uint32_t));
        cudaMemset(gpu_update_index->nbrs_[gamma_eidx], 0, (num_valid_vertices_buf_[batch] + 1) * sizeof(uint32_t*));
    }

    generateSizePtr<<<min((uint64_t)96 * 8, (el_update_dev.size() + 31) / 32), 32>>>(
        thrust::raw_pointer_cast(el_update_dev.data()),
        thrust::raw_pointer_cast(el_update_32_dev.data()),
        gpu_update_index->sizes_,
        gpu_update_index->nbrs_,
        el_update_dev.size());
    cErr(cudaDeviceSynchronize());
#ifdef ENABLE_CPU_DFS
    if (cpu_mirror_) {
        auto _t0 = std::chrono::high_resolution_clock::now();
        cpu_mirror_->RebuildUpdate(el_update_host_stage_[batch]);
        auto _t1 = std::chrono::high_resolution_clock::now();
        cpu_mirror_rebuild_ms_ += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
    }
#endif
    return true;
}

void CaLiGHelperGamma::PrintUpdateIndexSizes(uint32_t batch) {
    uint32_t num_v = num_valid_vertices_buf_[batch];
    if (num_v == 0) {
        std::cout << "[UpdateIndexSizes] No valid vertices in batch " << batch << std::endl;
        return;
    }
    std::vector<uint32_t> h_sizes(num_v + 1);
    std::cout << "[UpdateIndexSizes] batch=" << batch << " num_valid_vertices=" << num_v << std::endl;
    for (uint32_t gamma_eidx = 0; gamma_eidx < num_edges; gamma_eidx++) {
        cErr(cudaMemcpy(h_sizes.data(), gpu_update_index->sizes_[gamma_eidx],
                        (num_v + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint64_t total = 0;
        for (uint32_t v = 0; v <= num_v; v++) {
            total += h_sizes[v];
        }
        std::cout << "  edge " << gamma_eidx << ": " << total << " candidate edges" << std::endl;
    }
}
