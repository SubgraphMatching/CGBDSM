#include <cstdint>

#include "cub/cub.cuh"
#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/graph_gpu.h"

__forceinline__ __device__ smask_t computeSupportBit(uint32_t v0, uint32_t v1) {
    uint32_t h = v0 * 2654435761u ^ v1;
    h = (h >> 16) ^ h;
#if SUPPORT_MASK_WIDTH == 32
    return 1u << (h % 32u);
#elif SUPPORT_MASK_WIDTH == 64
    return 1ull << (h % 64u);
#endif
}


__global__ void setCapabilities(
    const Tries tries,
    uint32_t *capability
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < tries.vs_size_; i += num_threads)
    {
        capability[tries.vs_[i]] = tries.offs_[i + 1] - tries.offs_[i];
    }
}

__global__ void roundCapabilities(
    uint32_t* array,
    const uint32_t size
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += num_threads)
    {
        array[i] = max(MIN_NBR_SIZE, (uint32_t)exp2f(ceilf(log2f(array[i]) + 1)));
    }
}

__global__ void roundCapabilities(
    uint32_t *array,
    const uint32_t size,
    const uint32_t *label_array,
    const uint32_t q_label
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += num_threads)
        array[i] = label_array[i] == q_label ? max(MIN_NBR_SIZE, (uint32_t)exp2f(ceilf(log2f(array[i]) + 1))) : 0u;
}

__global__ void setNeighborPointers(
    uint32_t *base,
    const uint32_t *offsets,
    const uint32_t size,
    uint32_t **ptr
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += num_threads)
        ptr[i] = base + offsets[i];
}

__global__ void allocateFromMemPool(
    RelationsGPU data, const uint32_t idx,
    const uint32_t num_vertices, MemPool<uint32_t> pool
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;
    for (uint32_t v = tid; v < num_vertices; v += num_threads) {
        if (data.capability_[idx][v] > 0) {
            unsigned long long start = atomicAdd(pool.occupy_, (unsigned long long)data.capability_[idx][v]);
            data.nbrs_[idx][v] = pool.array_ + start;
        } else {
            data.nbrs_[idx][v] = nullptr;
        }
    }
}

__global__ void removeTriesFromGraph(
    const Tries del_tries,
    RelationsGPU data,
    const uint32_t idx
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;

    // One thread per vertex in del_tries
    for (uint32_t i = tid; i < del_tries.vs_size_; i += num_threads) {
        const uint32_t v = del_tries.vs_[i];
        const uint32_t* a = data.nbrs_[idx][v];                    // existing sorted neighbors
        const uint32_t a_size = data.sizes_[idx][v];
        const uint32_t* b = del_tries.nbrs_ + del_tries.offs_[i];  // sorted deletions
        const uint32_t b_size = del_tries.offs_[i + 1] - del_tries.offs_[i];

        // Sorted set difference: remove elements of b from a, write result in-place
        uint32_t* c = data.nbrs_[idx][v];
        uint32_t ai = 0, bi = 0, ci = 0;
        while (ai < a_size && bi < b_size) {
            if (a[ai] < b[bi]) {
                c[ci++] = a[ai++];
            } else if (a[ai] > b[bi]) {
                bi++;
            } else {
                // a[ai] == b[bi], skip (delete)
                ai++;
                bi++;
            }
        }
        // Copy remaining elements from a
        while (ai < a_size) {
            c[ci++] = a[ai++];
        }
        data.sizes_[idx][v] = ci;
    }
}

__global__ void addTriesToGraph(
    const Tries tries,
    RelationsGPU data,
    const uint32_t idx,
    MemPool<uint32_t> nbr_mem_pool
) {
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    const uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    __shared__ uint32_t insert_num[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t target_pos[NWARP_PER_BLOCK][WARP_SIZE];
    // Allocate WarpScan shared memory for 4 warps
    __shared__ typename cub::WarpScan<uint32_t>::TempStorage temp_storage[NWARP_PER_BLOCK];


    for (uint32_t i = gwarp_id; i < tries.vs_size_; i += num_warps)
    {
        if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capability_) return;

        const uint32_t& v = tries.vs_[i];

        const uint32_t* a = data.nbrs_[idx][v];
        const uint32_t* b = tries.nbrs_ + tries.offs_[i];
        uint32_t *c = data.nbrs_[idx][v];
        const uint32_t a_size = data.sizes_[idx][v];
        const uint32_t b_size = tries.offs_[i + 1] - tries.offs_[i];
        const uint32_t c_size = a_size + b_size;
        if (c_size >= data.capability_[idx][v])
        {
            if (lane_id == 0)
            {
                unsigned long long int new_capability = max((size_t)exp2(ceilf(log2f(c_size) + 1)), 8ul);
                unsigned long long int new_array_start = atomicAdd(nbr_mem_pool.occupy_, new_capability);
                if (new_array_start + new_capability < nbr_mem_pool.capability_)
                {
                    c = data.nbrs_[idx][v] = nbr_mem_pool.array_ + new_array_start;
                    data.capability_[idx][v] = new_capability;
                }
                else
                    printf("out of memory!\n");
            }
            c = (uint32_t*)__shfl_sync(0xffffffff, (unsigned long)c, 0);
        }
        __syncwarp();
        if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capability_) return;

        data.sizes_[idx][v] = c_size;

        uint32_t a_start = a_size - 1, a_end;
        uint32_t b_start = b_size - 1, b_end;
        uint32_t c_start = c_size - 1;

        while (a_start < a_size || b_start < b_size)
        {
            if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capability_) return;

            if (a_start >= a_size)
            {
                for (uint32_t j = b_start - lane_id; j < b_size; j -= WARP_SIZE)
                    c[j] = b[j];
                b_start = UINT32_MAX;
                continue;
            }
            if (b_start >= b_size)
            {
                for (uint32_t j = a_start - lane_id; j < a_size; j -= WARP_SIZE)
                    c[j] = a[j];
                a_start = UINT32_MAX;
                continue;
            }
            insert_num[warp_id][lane_id] = 0u;
            target_pos[warp_id][lane_id] = 0u;

            a_end = a_start + 1 - WARP_SIZE < a_size ? a_start + 1 - WARP_SIZE : 0;
            b_end = b_start + 1 - WARP_SIZE < b_size ? b_start + 1 - WARP_SIZE : 0;

            if (a[a_end] < b[b_end])
            {
                if (lane_id <= b_start - b_end)
                {
                    uint32_t insert_pos = lower_bound(a + a_end, a_start + 1 - a_end, b[b_start - lane_id]);
                    insert_pos = a_start + 1 - a_end - insert_pos;
                    //printf("find %d at %d\n", b[b_start - lane_id], insert_pos);
                    atomicAdd(&insert_num[warp_id][insert_pos], 1u);
                    target_pos[warp_id][lane_id] = insert_pos;
                }
                __syncwarp();
                cub::WarpScan<uint32_t>(temp_storage[warp_id]).ExclusiveSum(insert_num[warp_id][lane_id], insert_num[warp_id][lane_id]);

                bool write_a = true;
                if (lane_id <= a_start - a_end && a[a_start - lane_id] > b[b_end])
                {
                    //printf("write a c[%d]=%d\n", c_start - insert_num[warp_id][lane_id + 1] - lane_id, a[a_start - lane_id]);
                    write_a = false;
                    c[c_start - insert_num[warp_id][lane_id + 1] - lane_id] = a[a_start - lane_id];
                }
                __syncwarp();
                if (lane_id <= b_start - b_end)
                    c[c_start - target_pos[warp_id][lane_id] - lane_id] = b[b_start - lane_id];
                c_start = c_start - b_start + b_end - 1;
                b_start = b_end - 1;
                int next_first = __ffs(__ballot_sync(0xffffffff, write_a));
                __syncwarp();
                c_start = c_start - next_first + 1;
                a_start = a_start - next_first + 1;
            }
            else
            {
                if (lane_id <= a_start - a_end)
                {
                    uint32_t insert_pos = lower_bound(b + b_end, b_start + 1 - b_end, a[a_start - lane_id]);
                    insert_pos = b_start + 1 - b_end - insert_pos;
                    //printf("find %d at %d\n", a[a_start - lane_id], insert_pos);
                    atomicAdd(&insert_num[warp_id][insert_pos], 1u);
                    target_pos[warp_id][lane_id] = insert_pos;
                }
                __syncwarp();
                cub::WarpScan<uint32_t>(temp_storage[warp_id]).ExclusiveSum(insert_num[warp_id][lane_id], insert_num[warp_id][lane_id]);

                if (lane_id <= a_start - a_end)
                {
                    c[c_start - target_pos[warp_id][lane_id] - lane_id] = a[a_start - lane_id];
                }
                __syncwarp();
                bool write_a = true;
                if (lane_id <= b_start - b_end && b[b_start - lane_id] > a[a_end])
                {
                    write_a = false;
                    c[c_start - insert_num[warp_id][lane_id + 1] - lane_id] = b[b_start - lane_id];
                }
                c_start = c_start - a_start + a_end - 1;
                a_start = a_end - 1;
                int next_first = __ffs(__ballot_sync(0xffffffff, write_a));
                __syncwarp();
                c_start = c_start - next_first + 1;
                b_start = b_start - next_first + 1;
            }
        }
    }
}

// ============== Direct CSR Merge Kernels (no Tries intermediate) ==============

__global__ void setCapabilitiesFromSizes(
    uint32_t *sizes, uint32_t *capability, uint32_t n
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;
    for (uint32_t i = tid; i < n; i += num_threads)
        capability[i] = sizes[i];
}

__global__ void copyFromFlatCSR(
    uint32_t **old_nbrs, uint32_t *sizes,
    uint32_t **new_nbrs, uint32_t num_vertices
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;
    for (uint32_t u = tid; u < num_vertices; u += num_threads) {
        uint32_t sz = sizes[u];
        if (sz == 0 || new_nbrs[u] == nullptr) continue;
        for (uint32_t j = 0; j < sz; j++)
            new_nbrs[u][j] = old_nbrs[u][j];
    }
}

__global__ void mergeCSROntoGraph(
    RelationsGPU src,            // flat CSR (e.g. gpu_update_index)
    RelationsGPU dst,            // MemPool-backed GlobalIndex
    const uint32_t idx,          // gamma_eidx
    MemPool<uint32_t> nbr_mem_pool,
    uint32_t num_vertices        // DV_COUNT + 1
) {
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    const uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    __shared__ uint32_t insert_num[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t target_pos[NWARP_PER_BLOCK][WARP_SIZE];
    __shared__ typename cub::WarpScan<uint32_t>::TempStorage temp_storage[NWARP_PER_BLOCK];

    for (uint32_t u = gwarp_id; u < num_vertices; u += num_warps)
    {
        uint32_t b_size = src.sizes_[idx][u];
        if (b_size == 0) continue;

        if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capability_) return;

        const uint32_t* b = src.nbrs_[idx][u];
        const uint32_t* a = dst.nbrs_[idx][u];
        uint32_t *c = dst.nbrs_[idx][u];
        const uint32_t a_size = dst.sizes_[idx][u];
        const uint32_t c_size = a_size + b_size;
        if (c_size >= dst.capability_[idx][u])
        {
            if (lane_id == 0)
            {
                unsigned long long int new_capability = max((size_t)exp2(ceilf(log2f(c_size) + 1)), 8ul);
                unsigned long long int new_array_start = atomicAdd(nbr_mem_pool.occupy_, new_capability);
                if (new_array_start + new_capability < nbr_mem_pool.capability_)
                {
                    c = dst.nbrs_[idx][u] = nbr_mem_pool.array_ + new_array_start;
                    dst.capability_[idx][u] = new_capability;
                }
                else
                    printf("mergeCSROntoGraph: out of memory!\n");
            }
            c = (uint32_t*)__shfl_sync(0xffffffff, (unsigned long)c, 0);
        }
        __syncwarp();
        if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capability_) return;

        dst.sizes_[idx][u] = c_size;

        uint32_t a_start = a_size - 1, a_end;
        uint32_t b_start = b_size - 1, b_end;
        uint32_t c_start = c_size - 1;

        while (a_start < a_size || b_start < b_size)
        {
            if (*nbr_mem_pool.occupy_ >= nbr_mem_pool.capability_) return;

            if (a_start >= a_size)
            {
                for (uint32_t j = b_start - lane_id; j < b_size; j -= WARP_SIZE)
                    c[j] = b[j];
                b_start = UINT32_MAX;
                continue;
            }
            if (b_start >= b_size)
            {
                for (uint32_t j = a_start - lane_id; j < a_size; j -= WARP_SIZE)
                    c[j] = a[j];
                a_start = UINT32_MAX;
                continue;
            }
            insert_num[warp_id][lane_id] = 0u;
            target_pos[warp_id][lane_id] = 0u;

            a_end = a_start + 1 - WARP_SIZE < a_size ? a_start + 1 - WARP_SIZE : 0;
            b_end = b_start + 1 - WARP_SIZE < b_size ? b_start + 1 - WARP_SIZE : 0;

            if (a[a_end] < b[b_end])
            {
                if (lane_id <= b_start - b_end)
                {
                    uint32_t insert_pos = lower_bound(a + a_end, a_start + 1 - a_end, b[b_start - lane_id]);
                    insert_pos = a_start + 1 - a_end - insert_pos;
                    atomicAdd(&insert_num[warp_id][insert_pos], 1u);
                    target_pos[warp_id][lane_id] = insert_pos;
                }
                __syncwarp();
                cub::WarpScan<uint32_t>(temp_storage[warp_id]).ExclusiveSum(insert_num[warp_id][lane_id], insert_num[warp_id][lane_id]);

                bool write_a = true;
                if (lane_id <= a_start - a_end && a[a_start - lane_id] > b[b_end])
                {
                    write_a = false;
                    c[c_start - insert_num[warp_id][lane_id + 1] - lane_id] = a[a_start - lane_id];
                }
                __syncwarp();
                if (lane_id <= b_start - b_end)
                    c[c_start - target_pos[warp_id][lane_id] - lane_id] = b[b_start - lane_id];
                c_start = c_start - b_start + b_end - 1;
                b_start = b_end - 1;
                int next_first = __ffs(__ballot_sync(0xffffffff, write_a));
                __syncwarp();
                c_start = c_start - next_first + 1;
                a_start = a_start - next_first + 1;
            }
            else
            {
                if (lane_id <= a_start - a_end)
                {
                    uint32_t insert_pos = lower_bound(b + b_end, b_start + 1 - b_end, a[a_start - lane_id]);
                    insert_pos = b_start + 1 - b_end - insert_pos;
                    atomicAdd(&insert_num[warp_id][insert_pos], 1u);
                    target_pos[warp_id][lane_id] = insert_pos;
                }
                __syncwarp();
                cub::WarpScan<uint32_t>(temp_storage[warp_id]).ExclusiveSum(insert_num[warp_id][lane_id], insert_num[warp_id][lane_id]);

                if (lane_id <= a_start - a_end)
                {
                    c[c_start - target_pos[warp_id][lane_id] - lane_id] = a[a_start - lane_id];
                }
                __syncwarp();
                bool write_a = true;
                if (lane_id <= b_start - b_end && b[b_start - lane_id] > a[a_end])
                {
                    write_a = false;
                    c[c_start - insert_num[warp_id][lane_id + 1] - lane_id] = b[b_start - lane_id];
                }
                c_start = c_start - a_start + a_end - 1;
                a_start = a_end - 1;
                int next_first = __ffs(__ballot_sync(0xffffffff, write_a));
                __syncwarp();
                c_start = c_start - next_first + 1;
                b_start = b_start - next_first + 1;
            }
        }
    }
}

__global__ void removeCSROromGraph(
    RelationsGPU src,            // flat CSR with edges to delete
    RelationsGPU dst,            // GlobalIndex
    const uint32_t idx,
    uint32_t num_vertices
) {
    const uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t num_threads = blockDim.x * gridDim.x;
    for (uint32_t u = tid; u < num_vertices; u += num_threads) {
        uint32_t b_size = src.sizes_[idx][u];
        if (b_size == 0) continue;
        const uint32_t* b = src.nbrs_[idx][u];
        const uint32_t* a = dst.nbrs_[idx][u];
        uint32_t a_size = dst.sizes_[idx][u];
        uint32_t* c = dst.nbrs_[idx][u];
        uint32_t ai = 0, ci = 0;
        for (uint32_t bi = 0; bi < b_size; bi++) {
            while (ai < a_size && a[ai] < b[bi]) c[ci++] = a[ai++];
            if (ai < a_size && a[ai] == b[bi]) ai++;
        }
        while (ai < a_size) c[ci++] = a[ai++];
        dst.sizes_[idx][u] = ci;
    }
}

__global__ void statisticIndex(
    const RelationsGPU global_index,
    const uint32_t idx,
    uint32_t *sum,
    uint32_t *count
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < C_DV_COUNT; i += num_threads)
    {
        if (global_index.sizes_[idx][i] != 0)
        {
            atomicAdd(sum, global_index.sizes_[idx][i]);
            atomicAdd(count, 1u);
        }
    }
}

__global__ void getGlobalCandidates(
    const RelationsGPU data,
    const uint32_t idx,
    const Tries tries,
    uint32_t *cand_bits,
    bool *cand_flag,
    const uint32_t u
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < tries.vs_size_; i += num_threads)
    {
        uint32_t& dv = tries.vs_[i];
        bool pass = true;
        for (uint32_t j = C_QV_OFFS[u]; j < C_QV_OFFS[u + 1]; j++)
        {
            if ((j == idx ? data.sizes_[j][dv] + tries.offs_[i + 1] - tries.offs_[i] : data.sizes_[j][dv]) < C_NLF[j]) // Tries中存放的是更新的边的Tries结构，data是之前图的信息结构，这个遍历是为了判断更新前后顶点是否符合NLF，符合就返回True
            {
                pass = false;
                break;
            }
        }
        if (pass && ((cand_bits[dv / 32] & (1u << (dv % 32))) == 0))
        {
            atomicOr(&cand_bits[dv / 32], 1u << (dv % 32));
            cand_flag[i] = true;
        }
        __syncwarp();
    }
}

__global__ void getGlobalCandidateEdgesCount(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count
) {
    __shared__ uint32_t temp_num[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < new_cand_size; i += num_warps)
    {
        const uint32_t v = new_cand[i];
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        for (uint32_t j = 0; j < DIV_CEIL(data.sizes_[idx][v], WARP_SIZE); j++)
        {
            const uint32_t jj = j * WARP_SIZE + lane_id;
            const uint32_t warp_sum = __popc(__ballot_sync(
                0xffffffff, 
                jj < data.sizes_[idx][v] &&
                (other_cand_bits[data.nbrs_[idx][v][jj] / 32] & (1 << (data.nbrs_[idx][v][jj] % 32))) > 0
            ));
            if (lane_id == 0u)
            {
                atomicAdd(&temp_num[warp_id], warp_sum);
            }
            __syncwarp();
        }
        __syncwarp();
        if (lane_id == 0) cand_e_count[i] = temp_num[warp_id];
        __syncwarp();
    }
}

__global__ void getGlobalCandidateEdgesWrite(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count_prefix_sum,
    uint32_t *relation_u,
    uint32_t *relation_uu
) {
    __shared__ uint32_t write_pos[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < new_cand_size; i += num_warps)
    {
        const uint32_t& v = new_cand[i];
        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        for (uint32_t j = 0; j < DIV_CEIL(data.sizes_[idx][v], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < data.sizes_[idx][v])
            {
                nbr = data.nbrs_[idx][v][j * WARP_SIZE + lane_id];
                if ((other_cand_bits[nbr / 32] & (1 << (nbr % 32))) > 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[cand_e_count_prefix_sum[i] + write_pos[warp_id] + rank] = v;
                relation_uu[cand_e_count_prefix_sum[i] + write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void filterRelevantCount(
    const Tries input,
    Tries output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t temp_num[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            output.offs_[i] = 0u;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                const uint32_t jj = j * WARP_SIZE + lane_id;
                const uint32_t warp_sum = __popc(__ballot_sync(
                    0xffffffff, 
                    jj < input.offs_[i + 1] - input.offs_[i] &&
                    (second_candidates[input.nbrs_[jj + input.offs_[i]] / 32] & 1 << (input.nbrs_[jj + input.offs_[i]] % 32)) > 0
                ));
                if (lane_id == 0u)
                {
                    atomicAdd(&temp_num[warp_id], warp_sum);
                }
                __syncwarp();
            }
            __syncwarp();
            if (lane_id == 0) output.offs_[i] = temp_num[warp_id];
            __syncwarp();
        }
    }
}

__global__ void filterRelevantWrite(
    const Tries input,
    Tries output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t write_pos[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            continue;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                bool found = false;
                if (j * WARP_SIZE + lane_id < input.offs_[i + 1] - input.offs_[i])
                {
                    nbr = input.nbrs_[j * WARP_SIZE + lane_id + input.offs_[i]];
                    if ((second_candidates[nbr / 32] & 1 << (nbr % 32)) > 0)
                    {
                        found = true;
                    }
                }
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                if (found)
                {
                    const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                    output.nbrs_[output.offs_[i] + write_pos[warp_id] + rank] = nbr;
                    if (rank == 0)
                    {
                        write_pos[warp_id] += __popc(found_mask);
                    }
                }
                __syncwarp();
            }
            __syncwarp();
        }
    }
}

// for building local index

__global__ void edgeList2RelationCount(
    const Tries input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t temp_num[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            output.sizes_[idx][dv] = 0u;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                const uint32_t jj = j * WARP_SIZE + lane_id;
                const uint32_t warp_sum = __popc(__ballot_sync(
                    0xffffffff, 
                    jj < input.offs_[i + 1] - input.offs_[i] &&
                    (second_candidates[input.nbrs_[jj + input.offs_[i]] / 32] & 1 << (input.nbrs_[jj + input.offs_[i]] % 32)) > 0
                ));
                if (lane_id == 0u)
                {
                    atomicAdd(&temp_num[warp_id], warp_sum);
                }
                __syncwarp();
            }
            __syncwarp();
            if (lane_id == 0) output.sizes_[idx][dv] = temp_num[warp_id];
            __syncwarp();
        }
    }
}

__global__ void edgeList2RelationWrite(
    const Tries input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
) {
    __shared__ uint32_t write_pos[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t i = gwarp_id; i < input.vs_size_; i += num_warps)
    {
        uint32_t& dv = input.vs_[i];
        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        if ((first_candidates[dv / 32] & 1u << (dv % 32)) == 0)
        {
            continue;
        }
        else
        {
            for (uint32_t j = 0; j < DIV_CEIL(input.offs_[i + 1] - input.offs_[i], WARP_SIZE); j++)
            {
                bool found = false;
                if (j * WARP_SIZE + lane_id < input.offs_[i + 1] - input.offs_[i])
                {
                    nbr = input.nbrs_[j * WARP_SIZE + lane_id + input.offs_[i]];
                    if ((second_candidates[nbr / 32] & 1 << (nbr % 32)) > 0)
                    {
                        found = true;
                    }
                }
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                if (found)
                {
                    const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                    output.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                    if (rank == 0)
                    {
                        write_pos[warp_id] += __popc(found_mask);
                    }
                }
                __syncwarp();
            }
            __syncwarp();
        }
    }
}

// for building local index v2
__global__ void getLocalCandidatesBackward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t found[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[backward_index][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) found[warp_id] = false;
        __syncwarp();

        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[backward_index][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[backward_index][dv])
            {
                nbr = global_index.nbrs_[backward_index][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (local_index.sizes_[first_bn_of_uu_index][nbr] != 0)
                {
                    atomicOr(&found[warp_id], 1u);
                }
            }
            __syncwarp();
            if (found[warp_id]) break;
        }

        if (found[warp_id])
        {
            if (lane_id == 0) cum_bn[dv] += 1u;
            __syncwarp();
        }
    }
}

__global__ void getLocalCandidatesForward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t forward_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (local_index.sizes_[first_bn_of_uu_index][dv] == 0u)
        {
            continue;
        }
        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[forward_index][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[forward_index][dv])
            {
                auto nbr = global_index.nbrs_[forward_index][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (global_index.sizes_[backward_index][nbr] != 0u && cum_bn[nbr] == pre_max_cum)
                {
                    //atomicMax(&cum_bn[nbr], pre_max_cum + 1u);
                    atomicCAS(&cum_bn[nbr], pre_max_cum, pre_max_cum + 1u);
                }
            }
            __syncwarp();
        }
    }
}

__global__ void buildLocalRelationNew2OldCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t temp_num[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[idx][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            local_index.sizes_[idx][dv] = 0u;
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        // a warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (local_index.sizes_[first_bn_of_uu_index][nbr] != 0)
                {
                    atomicAdd(&temp_num[warp_id], 1u);
                }
            }
            __syncwarp();
        }

        local_index.sizes_[idx][dv] = temp_num[warp_id];
    }
}

__global__ void buildLocalRelationNew2OldWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
) {
    __shared__ uint32_t write_pos[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[idx][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (local_index.sizes_[first_bn_of_uu_index][nbr] != 0)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[local_index.capability_[idx][dv] + write_pos[warp_id] + rank] = dv;
                local_index.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void buildLocalRelationOld2NewCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t temp_num[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (local_index.sizes_[first_bn_of_uu_index][dv] == 0u)
        {
            local_index.sizes_[idx][dv] = 0u;
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) temp_num[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (cum_bn[nbr] == pre_max_cum)
                {
                    atomicAdd(&temp_num[warp_id], 1u);
                }
            }
            __syncwarp();
        }

        local_index.sizes_[idx][dv] = temp_num[warp_id];
    }
}

__global__ void buildLocalRelationOld2NewWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
) {
    __shared__ uint32_t write_pos[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (local_index.sizes_[first_bn_of_uu_index][dv] == 0u)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) write_pos[warp_id] = 0u;
        __syncwarp();

        // each warp check all neighbors of dv
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[idx][dv], WARP_SIZE); j++)
        {
            bool found = false;
            if (j * WARP_SIZE + lane_id < global_index.sizes_[idx][dv])
            {
                nbr = global_index.nbrs_[idx][dv][j * WARP_SIZE + lane_id];
                // if the neighbor is in the local index
                if (cum_bn[nbr] == pre_max_cum)
                {
                    found = true;
                }
            }
            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
            if (found)
            {
                const uint32_t rank = lane_id == 0 ? 0 : __popc((UINT32_MAX >> (WARP_SIZE - lane_id)) & found_mask);
                relation_u[local_index.capability_[idx][dv] + write_pos[warp_id] + rank] = dv;
                local_index.nbrs_[idx][dv][write_pos[warp_id] + rank] = nbr;
                if (rank == 0)
                {
                    write_pos[warp_id] += __popc(found_mask);
                }
            }
            __syncwarp();
        }
        __syncwarp();
    }
}

__global__ void mapTrieToRelation(
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *vs,
    const uint32_t *offs,
    const uint32_t num_items
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < num_items; i += num_threads)
    {
        local_index.sizes_[idx][vs[i]] = offs[i];
    }
}

// ============== Valid Bit Kernels ==============

__global__ void setLocalCandidateValidBits(
    const uint32_t *cum_bn,
    const uint32_t max_cum,
    const uint8_t qv_idx
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t dv = tid; dv < C_DV_COUNT; dv += num_threads)
    {
        if (cum_bn[dv] == max_cum)
        {
            atomicOr(&C_VALID_BITS.bits_[qv_idx][dv / 32u], 1u << (dv % 32u));
        }
    }
}

__global__ void setInitialValidBits(
    const RelationsGPU index,
    const uint8_t idx,
    const uint8_t qv_idx
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;

    for (uint32_t dv = tid; dv < C_DV_COUNT; dv += num_threads)
    {
        if (index.sizes_[idx][dv] > 0u)
        {
            atomicOr(&C_VALID_BITS.bits_[qv_idx][dv / 32u], 1u << (dv % 32u));
        }
    }
}

__global__ void getLocalCandidatesBackwardBit(
    const RelationsGPU global_index,
    const uint8_t backward_index,
    const uint8_t bn_qv,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    __shared__ uint32_t found[NWARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if (global_index.sizes_[backward_index][dv] == 0u || cum_bn[dv] != pre_max_cum)
        {
            continue;
        }

        uint32_t nbr = UINT32_MAX;
        if (lane_id == 0) found[warp_id] = false;
        __syncwarp();

        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[backward_index][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[backward_index][dv])
            {
                nbr = global_index.nbrs_[backward_index][dv][j * WARP_SIZE + lane_id];
                if ((C_VALID_BITS.bits_[bn_qv][nbr / 32u] & (1u << (nbr % 32u))) != 0u)
                {
                    atomicOr(&found[warp_id], 1u);
                }
            }
            __syncwarp();
            if (found[warp_id]) break;
        }

        if (found[warp_id])
        {
            if (lane_id == 0) cum_bn[dv] += 1u;
            __syncwarp();
        }
    }
}

__global__ void getLocalCandidatesForwardBit(
    const RelationsGPU global_index,
    const uint8_t forward_index,
    const uint8_t backward_index,
    const uint8_t bn_qv,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
) {
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t gwarp_id = warp_id + blockDim.x * blockIdx.x / WARP_SIZE;
    uint32_t num_warps = blockDim.x * gridDim.x / WARP_SIZE;

    for (uint32_t dv = gwarp_id; dv < C_DV_COUNT; dv += num_warps)
    {
        if ((C_VALID_BITS.bits_[bn_qv][dv / 32u] & (1u << (dv % 32u))) == 0u)
        {
            continue;
        }
        for (uint32_t j = 0; j < DIV_CEIL(global_index.sizes_[forward_index][dv], WARP_SIZE); j++)
        {
            if (j * WARP_SIZE + lane_id < global_index.sizes_[forward_index][dv])
            {
                auto nbr = global_index.nbrs_[forward_index][dv][j * WARP_SIZE + lane_id];
                if (global_index.sizes_[backward_index][nbr] != 0u && cum_bn[nbr] == pre_max_cum)
                {
                    atomicCAS(&cum_bn[nbr], pre_max_cum, pre_max_cum + 1u);
                }
            }
            __syncwarp();
        }
    }
}

// ============== Merged Valid Bit Kernels (per-edge, dual-source) ==============

__global__ void setInitialValidBitsAll(
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nt = gridDim.x * blockDim.x;

    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[0];
    uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[1];
    uint8_t i01 = C_EIDX[u0 * C_QV_COUNT + u1];
    uint8_t i10 = C_EIDX[u1 * C_QV_COUNT + u0];

    for (uint32_t dv = tid; dv < C_DV_COUNT; dv += nt) {
        smask_t sm_u0 = SMASK_ZERO, sz01 = update_index.sizes_[i01][dv];
        for (uint32_t j = 0; j < sz01; j++) {
            uint32_t v1 = update_index.nbrs_[i01][dv][j];
            sm_u0 |= computeSupportBit(dv, v1);
        }
        if (sm_u0 != SMASK_ZERO)
            atomicOr(&d_edge_sm_ptrs[ei * MAX_VCOUNT + u0][dv], sm_u0);

        smask_t sm_u1 = SMASK_ZERO, sz10 = update_index.sizes_[i10][dv];
        for (uint32_t j = 0; j < sz10; j++) {
            uint32_t v0 = update_index.nbrs_[i10][dv][j];
            sm_u1 |= computeSupportBit(v0, dv);
        }
        if (sm_u1 != SMASK_ZERO)
            atomicOr(&d_edge_sm_ptrs[ei * MAX_VCOUNT + u1][dv], sm_u1);
    }
}

__global__ void checkAllConstraintsAndSetBitsAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[depth];

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    auto bn_s = C_INDEXING_ORDERS[ei].bni_offs_[depth];
    auto bn_e = C_INDEXING_ORDERS[ei].bni_offs_[depth + 1];
    uint8_t num_bn = bn_e - bn_s;

    for (uint32_t dv = gwid; dv < C_DV_COUNT; dv += nwarp) {
        smask_t combined_mask = SMASK_ALL;
        for (uint8_t bp = 0; bp < num_bn && combined_mask != SMASK_ZERO; bp++) {
            uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[C_INDEXING_ORDERS[ei].bni_[bn_s + bp]];
            uint8_t bwd = C_EIDX[u0 * C_QV_COUNT + u1];
            smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1];
            bool up_vis = (C_DIR_TO_EDGE[bwd] <= ei);

            smask_t bn_mask = SMASK_ZERO;
            if (index_gpu.sizes_[bwd][dv] > 0u) {
                for (uint32_t j = 0; j < DIV_CEIL(index_gpu.sizes_[bwd][dv], WARP_SIZE); j++) {
                    uint32_t nb = UINT32_MAX;
                    if (j * WARP_SIZE + lid < index_gpu.sizes_[bwd][dv])
                        nb = index_gpu.nbrs_[bwd][dv][j * WARP_SIZE + lid];
                    smask_t nb_sm = (nb != UINT32_MAX) ? sm_u1[nb] : SMASK_ZERO;
                    for (uint32_t d = 16; d > 0; d >>= 1)
                        nb_sm |= __shfl_down_sync(0xffffffff, nb_sm, d);
                    bn_mask |= __shfl_sync(0xffffffff, nb_sm, 0);
                    if ((bn_mask & combined_mask) == combined_mask) break;
                }
            }

            if (up_vis && update_index.sizes_[bwd][dv] > 0u) {
                for (uint32_t j = 0; j < DIV_CEIL(update_index.sizes_[bwd][dv], WARP_SIZE); j++) {
                    uint32_t nb = UINT32_MAX;
                    if (j * WARP_SIZE + lid < update_index.sizes_[bwd][dv])
                        nb = update_index.nbrs_[bwd][dv][j * WARP_SIZE + lid];
                    smask_t nb_sm = (nb != UINT32_MAX) ? sm_u1[nb] : SMASK_ZERO;
                    for (uint32_t d = 16; d > 0; d >>= 1)
                        nb_sm |= __shfl_down_sync(0xffffffff, nb_sm, d);
                    bn_mask |= __shfl_sync(0xffffffff, nb_sm, 0);
                    if ((bn_mask & combined_mask) == combined_mask) break;
                }
            }
            combined_mask &= bn_mask;
        }

        if (combined_mask != SMASK_ZERO && lid == 0)
            atomicOr(&d_edge_sm_ptrs[ei * MAX_VCOUNT + u0][dv], combined_mask);
    }
}

__global__ void pushFromFirstBNAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[depth];

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    auto bn_s = C_INDEXING_ORDERS[ei].bni_offs_[depth];
    uint8_t bn_depth = C_INDEXING_ORDERS[ei].bni_[bn_s];
    uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[bn_depth];

    // Forward direction: u1 -> u0 (reverse of the backward edge u0 -> u1)
    uint8_t fwd = C_EIDX[u1 * C_QV_COUNT + u0];
    bool up_vis = (C_DIR_TO_EDGE[fwd] <= ei);

    smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1];
    smask_t* sm_u0 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u0];

    for (uint32_t nb = gwid; nb < C_DV_COUNT; nb += nwarp) {
        smask_t nb_mask = sm_u1[nb];
        if (nb_mask == SMASK_ZERO) continue;

        // Push to forward neighbors of nb along u1 -> u0 direction
        if (index_gpu.sizes_[fwd][nb] > 0u) {
            for (uint32_t j = lid; j < index_gpu.sizes_[fwd][nb]; j += WARP_SIZE) {
                uint32_t dv = index_gpu.nbrs_[fwd][nb][j];
                atomicOr(&sm_u0[dv], nb_mask);
            }
        }

        if (up_vis && update_index.sizes_[fwd][nb] > 0u) {
            for (uint32_t j = lid; j < update_index.sizes_[fwd][nb]; j += WARP_SIZE) {
                uint32_t dv = update_index.nbrs_[fwd][nb][j];
                atomicOr(&sm_u0[dv], nb_mask);
            }
        }
    }
}

__global__ void verifyCandidatesAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[depth];

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    auto bn_s = C_INDEXING_ORDERS[ei].bni_offs_[depth];
    auto bn_e = C_INDEXING_ORDERS[ei].bni_offs_[depth + 1];
    uint8_t num_bn = bn_e - bn_s;

    smask_t* sm_u0 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u0];

    for (uint32_t dv = gwid; dv < C_DV_COUNT; dv += nwarp) {
        smask_t combined_mask = sm_u0[dv];
        if (combined_mask == SMASK_ZERO) continue;

        // Check remaining backward neighbors (skip first BN, already handled by Push)
        for (uint8_t bp = 1; bp < num_bn && combined_mask != SMASK_ZERO; bp++) {
            uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[C_INDEXING_ORDERS[ei].bni_[bn_s + bp]];
            uint8_t bwd = C_EIDX[u0 * C_QV_COUNT + u1];
            smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1];
            bool up_vis = (C_DIR_TO_EDGE[bwd] <= ei);

            smask_t bn_mask = SMASK_ZERO;
            if (index_gpu.sizes_[bwd][dv] > 0u) {
                for (uint32_t j = 0; j < DIV_CEIL(index_gpu.sizes_[bwd][dv], WARP_SIZE); j++) {
                    uint32_t nb = UINT32_MAX;
                    if (j * WARP_SIZE + lid < index_gpu.sizes_[bwd][dv])
                        nb = index_gpu.nbrs_[bwd][dv][j * WARP_SIZE + lid];
                    smask_t nb_sm = (nb != UINT32_MAX) ? sm_u1[nb] : SMASK_ZERO;
                    for (uint32_t d = 16; d > 0; d >>= 1)
                        nb_sm |= __shfl_down_sync(0xffffffff, nb_sm, d);
                    bn_mask |= __shfl_sync(0xffffffff, nb_sm, 0);
                    if ((bn_mask & combined_mask) == combined_mask) break;
                }
            }

            if (up_vis && update_index.sizes_[bwd][dv] > 0u) {
                for (uint32_t j = 0; j < DIV_CEIL(update_index.sizes_[bwd][dv], WARP_SIZE); j++) {
                    uint32_t nb = UINT32_MAX;
                    if (j * WARP_SIZE + lid < update_index.sizes_[bwd][dv])
                        nb = update_index.nbrs_[bwd][dv][j * WARP_SIZE + lid];
                    smask_t nb_sm = (nb != UINT32_MAX) ? sm_u1[nb] : SMASK_ZERO;
                    for (uint32_t d = 16; d > 0; d >>= 1)
                        nb_sm |= __shfl_down_sync(0xffffffff, nb_sm, d);
                    bn_mask |= __shfl_sync(0xffffffff, nb_sm, 0);
                    if ((bn_mask & combined_mask) == combined_mask) break;
                }
            }
            combined_mask &= bn_mask;
        }

        if (lid == 0)
            sm_u0[dv] = combined_mask;
    }
}

__global__ void pushAllBackwardBNAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs,
    smask_t** d_edge_sm_tmp_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[depth];

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    auto bn_s = C_INDEXING_ORDERS[ei].bni_offs_[depth];
    auto bn_e = C_INDEXING_ORDERS[ei].bni_offs_[depth + 1];
    uint8_t num_bn = bn_e - bn_s;

    // Outer loop over data vertices (scanned once), inner loop over backward neighbors.
    // Enables a vertex-level pre-check: skip nb that is inactive for ALL backward neighbors.
    for (uint32_t nb = gwid; nb < C_DV_COUNT; nb += nwarp) {
        // Phase 1: pre-check — load masks for all backward neighbors, test if any is active
        smask_t nb_masks[MAX_VCOUNT];
        bool any_active = false;
        for (uint8_t bp = 0; bp < num_bn; bp++) {
            uint8_t bn_depth = C_INDEXING_ORDERS[ei].bni_[bn_s + bp];
            uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[bn_depth];
            smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1];
            nb_masks[bp] = sm_u1[nb];
            if (nb_masks[bp] != SMASK_ZERO) any_active = true;
        }
        if (!any_active) continue;   // fully inactive — skip the costly edge traversal

        // Phase 2: push along each active backward neighbor's forward direction (u1 -> u0)
        for (uint8_t bp = 0; bp < num_bn; bp++) {
            smask_t nb_mask = nb_masks[bp];
            if (nb_mask == SMASK_ZERO) continue;

            uint8_t bn_depth = C_INDEXING_ORDERS[ei].bni_[bn_s + bp];
            uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[bn_depth];
            // Forward direction: u1 -> u0 (reverse of the backward edge u0 -> u1)
            uint8_t fwd = C_EIDX[u1 * C_QV_COUNT + u0];
            bool up_vis = (C_DIR_TO_EDGE[fwd] <= ei);
            smask_t* tmp_u1 = d_edge_sm_tmp_ptrs[ei * MAX_VCOUNT + u1];

            if (index_gpu.sizes_[fwd][nb] > 0u) {
                for (uint32_t j = lid; j < index_gpu.sizes_[fwd][nb]; j += WARP_SIZE) {
                    uint32_t dv = index_gpu.nbrs_[fwd][nb][j];
                    if(nb_mask & tmp_u1[dv] == tmp_u1[dv])
                        continue;
                    atomicOr(&tmp_u1[dv], nb_mask);
                }
            }

            if (up_vis && update_index.sizes_[fwd][nb] > 0u) {
                for (uint32_t j = lid; j < update_index.sizes_[fwd][nb]; j += WARP_SIZE) {
                    uint32_t dv = update_index.nbrs_[fwd][nb][j];
                    if(nb_mask & tmp_u1[dv] == tmp_u1[dv])
                        continue;
                    atomicOr(&tmp_u1[dv], nb_mask);
                }
            }
        }
    }
}

__global__ void intersectAndSetBitsAll(
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs,
    smask_t** d_edge_sm_tmp_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[depth];

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    auto bn_s = C_INDEXING_ORDERS[ei].bni_offs_[depth];
    auto bn_e = C_INDEXING_ORDERS[ei].bni_offs_[depth + 1];
    uint8_t num_bn = bn_e - bn_s;

    smask_t* sm_u0 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u0];

    for (uint32_t dv = gwid; dv < C_DV_COUNT; dv += nwarp) {
        // Phase 1: warp-cooperative AND-reduce. num_bn <= QV_COUNT-1 <= 11 < WARP_SIZE,
        // so lane lid reads the temp value for bp=lid; lanes >= num_bn contribute SMASK_ALL
        // (identity for AND), then __shfl_down_sync reduces across the warp.
        smask_t my_tu = SMASK_ALL;
        if (lid < num_bn) {
            uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[C_INDEXING_ORDERS[ei].bni_[bn_s + lid]];
            my_tu = d_edge_sm_tmp_ptrs[ei * MAX_VCOUNT + u1][dv];
        }
        for (uint32_t d = 16; d > 0; d >>= 1)
            my_tu &= __shfl_down_sync(0xffffffff, my_tu, d);
        smask_t combined = __shfl_sync(0xffffffff, my_tu, 0);  // broadcast to all lanes

        // Phase 2: warp-cooperative clear — lane lid clears bp=lid's temp (unconditional,
        // prevents stale bits leaking into the next depth)
        if (lid < num_bn) {
            uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[C_INDEXING_ORDERS[ei].bni_[bn_s + lid]];
            d_edge_sm_tmp_ptrs[ei * MAX_VCOUNT + u1][dv] = SMASK_ZERO;
        }

        // Phase 3: lane 0 writes the result (sm_u0 was zero before this depth)
        if (lid == 0 && combined != SMASK_ZERO)
            sm_u0[dv] = combined;
    }
}

__global__ void forwardLookaheadAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[depth];
    smask_t* sm_u0 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u0];

    auto fn_s = C_INDEXING_ORDERS_EXT[ei].fni_offs_[depth];
    auto fn_e = C_INDEXING_ORDERS_EXT[ei].fni_offs_[depth + 1];
    if (fn_s == fn_e) return;

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    for (uint32_t dv = gwid; dv < C_DV_COUNT; dv += nwarp) {
        smask_t dv_sm = sm_u0[dv];
        if (dv_sm == SMASK_ZERO) continue;

        smask_t fwd_mask = SMASK_ALL;

        for (uint8_t fp = fn_s; fp < fn_e && fwd_mask != SMASK_ZERO; fp++) {
            uint8_t fn_depth = C_INDEXING_ORDERS_EXT[ei].fni_[fp];
            uint8_t u_f = C_INDEXING_ORDERS[ei].vs_[fn_depth];
            uint8_t fwd_idx = C_INDEXING_ORDERS_EXT[ei].fni_eidx_[fp];
            smask_t* sm_uf = d_edge_sm_ptrs[ei * MAX_VCOUNT + u_f];
            bool up_vis = (C_DIR_TO_EDGE[fwd_idx] <= ei);

            smask_t fn_mask = SMASK_ZERO;

            if (index_gpu.sizes_[fwd_idx][dv] > 0u) {
                for (uint32_t j = 0; j < DIV_CEIL(index_gpu.sizes_[fwd_idx][dv], WARP_SIZE); j++) {
                    uint32_t nb = UINT32_MAX;
                    if (j * WARP_SIZE + lid < index_gpu.sizes_[fwd_idx][dv])
                        nb = index_gpu.nbrs_[fwd_idx][dv][j * WARP_SIZE + lid];
                    smask_t nb_sm = (nb != UINT32_MAX) ? sm_uf[nb] : SMASK_ZERO;
                    for (uint32_t d = 16; d > 0; d >>= 1)
                        nb_sm |= __shfl_down_sync(0xffffffff, nb_sm, d);
                    fn_mask |= __shfl_sync(0xffffffff, nb_sm, 0);
                    if ((fn_mask & fwd_mask) == fwd_mask) break;
                }
            }

            if (up_vis && update_index.sizes_[fwd_idx][dv] > 0u) {
                for (uint32_t j = 0; j < DIV_CEIL(update_index.sizes_[fwd_idx][dv], WARP_SIZE); j++) {
                    uint32_t nb = UINT32_MAX;
                    if (j * WARP_SIZE + lid < update_index.sizes_[fwd_idx][dv])
                        nb = update_index.nbrs_[fwd_idx][dv][j * WARP_SIZE + lid];
                    smask_t nb_sm = (nb != UINT32_MAX) ? sm_uf[nb] : SMASK_ZERO;
                    for (uint32_t d = 16; d > 0; d >>= 1)
                        nb_sm |= __shfl_down_sync(0xffffffff, nb_sm, d);
                    fn_mask |= __shfl_sync(0xffffffff, nb_sm, 0);
                    if ((fn_mask & fwd_mask) == fwd_mask) break;
                }
            }

            fwd_mask &= fn_mask;
        }

        smask_t new_sm = dv_sm & fwd_mask;
        if (new_sm == SMASK_ZERO && lid == 0)
            sm_u0[dv] = SMASK_ZERO;
        else if (new_sm != dv_sm && lid == 0)
            sm_u0[dv] = new_sm;
    }
}

__global__ void shallowDFSExpand(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs,
    const uint8_t end_depth
) {
    uint8_t ei = blockIdx.y;
    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[0];
    uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[1];
    uint8_t idx01 = C_EIDX[u0 * C_QV_COUNT + u1];
    // sm_u0/sm_u1: support mask arrays for query vertices u0/u1
    smask_t* sm_u0 = d_edge_sm_ptrs[ei * MAX_VCOUNT +u0];
    smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT +u1];

    // Shared memory DFS stack (same layout as extendBFSDFSRegTwoBit)
    __shared__ uint32_t result_queue[NWARP_PER_BLOCK][MAX_VCOUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ bool intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t dfs_depth[NWARP_PER_BLOCK];
    __shared__ uint32_t compact_nbrs[NWARP_PER_BLOCK][WARP_SIZE * 2];
    __shared__ uint32_t compact_pos[NWARP_PER_BLOCK][WARP_SIZE * 2];
    __shared__ uint8_t compact_count[NWARP_PER_BLOCK];
    // Per-warp path masks for support mask tracking through DFS depths
    // path_masks[wid][0] = base mask (sm_u0[v0] & sm_u1[v1])
    // path_masks[wid][d-1] = base & sm[u2][v2] & ... & sm[u_{d}][v_{d}] for depth d >= 2
    __shared__ smask_t path_masks[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    // Each warp processes one valid v0
    for (uint32_t v0 = gwid; v0 < C_DV_COUNT; v0 += nwarp) {
        if (sm_u0[v0] == SMASK_ZERO) continue;
        uint32_t up_sz = update_index.sizes_[idx01][v0];
        for (uint32_t j = 0; j < up_sz; j += WARP_SIZE) {
            uint32_t v1 = update_index.nbrs_[idx01][v0][j + lid];
            bool v1_valid = (v1 != UINT32_MAX) && (sm_u1[v1] != SMASK_ZERO);

            uint32_t valid_mask = __ballot_sync(0xffffffff, v1_valid);

            // Process valid v1 candidates one at a time (serialized within warp)
            while (valid_mask != 0u) {
                uint8_t bit_pos = __ffs(valid_mask) - 1;
                valid_mask &= (valid_mask - 1u);
                uint32_t cur_v1 = __shfl_sync(0xffffffff, v1, bit_pos);

                // Compute base path mask: AND of support masks for v0 and v1
                smask_t pm_v0 = sm_u0[v0];
                smask_t pm_v1 = sm_u1[cur_v1];
                smask_t base_pm = pm_v0 & pm_v1;
                if (base_pm == SMASK_ZERO) continue; // no common support bit

                // Initialize DFS state
                if (lid == 0) dfs_depth[wid] = 2;
                if (lid < MAX_VCOUNT - 2) {
                    queue_pos[wid][lid] = 0u;
                    queue_size[wid][lid] = 0u;
                    end_v[wid][lid] = 0u;
                    end_nbr[wid][lid] = 0u;
                    intersection_continue[wid][lid] = false;
                }
                if (lid == 0) compact_count[wid] = 0;
                // Initialize path_masks: slot 0 = base mask (v0 & v1)
                if (lid < MAX_VCOUNT - 2) path_masks[wid][lid] = SMASK_ZERO;
                if (lid == 0) path_masks[wid][0] = base_pm;
                __syncwarp();

                // DFS main loop
                while (dfs_depth[wid] >= 2) {
                    __syncwarp();

                    const uint8_t& pre_qv_idx =
                        C_INDEXING_ORDERS[ei].bni_[C_INDEXING_ORDERS[ei].bni_offs_[dfs_depth[wid]]];
                    const uint8_t& pre_qe_idx =
                        C_EIDX[C_INDEXING_ORDERS[ei].vs_[pre_qv_idx] * C_QV_COUNT +
                                C_INDEXING_ORDERS[ei].vs_[dfs_depth[wid]]];

                    if (queue_pos[wid][dfs_depth[wid] - 2] >= queue_size[wid][dfs_depth[wid] - 2])
                    {
                        // Queue exhausted: backtrack or continue scanning
                        if (intersection_continue[wid][dfs_depth[wid] - 2] &&
                            ((pre_qv_idx < 2 && end_v[wid][dfs_depth[wid] - 2] > 0) ||
                             (pre_qv_idx >= 2 && end_v[wid][dfs_depth[wid] - 2] >
                              queue_pos[wid][pre_qv_idx - 2])))
                        {
                            if (lid == 0) {
                                queue_pos[wid][dfs_depth[wid] - 2] = 0u;
                                queue_size[wid][dfs_depth[wid] - 2] = 0u;
                                intersection_continue[wid][dfs_depth[wid] - 2] = false;
                                dfs_depth[wid]--;
                                if (dfs_depth[wid] >= 2)
                                    queue_pos[wid][dfs_depth[wid] - 2]++;
                            }
                            __syncwarp();
                        }
                        else
                        {
                            // Scan new candidates
                            if (!intersection_continue[wid][dfs_depth[wid] - 2]) {
                                if (lid == 0) {
                                    if (pre_qv_idx < 2) {
                                        end_v[wid][dfs_depth[wid] - 2] = 0u;
                                        end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                                        compact_count[wid] = 0;
                                    } else {
                                        end_v[wid][dfs_depth[wid] - 2] =
                                            queue_pos[wid][pre_qv_idx - 2];
                                        end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                                        compact_count[wid] = 0;
                                    }
                                }
                                __syncwarp();
                            }
                            intersection_continue[wid][dfs_depth[wid] - 2] = true;

                            // Get predecessor data vertex
                            uint32_t pre_dv = UINT32_MAX;
                            if (pre_qv_idx == 0u) pre_dv = v0;
                            else if (pre_qv_idx == 1u) pre_dv = cur_v1;
                            else pre_dv = result_queue[wid][pre_qv_idx - 2]
                                              [queue_pos[wid][pre_qv_idx - 2]];

                            uint8_t current_qv = C_INDEXING_ORDERS[ei].vs_[dfs_depth[wid]];

                            // Dual-source neighbor read + compaction
                            uint32_t read_end = index_gpu.sizes_[pre_qe_idx][pre_dv];
                            bool up_vis2 = (C_DIR_TO_EDGE[pre_qe_idx] < ei);
                            uint32_t up_read_sz = up_vis2
                                ? update_index.sizes_[pre_qe_idx][pre_dv] : 0u;
                            uint32_t total_read = read_end + up_read_sz;

                            uint32_t read_offset = end_nbr[wid][dfs_depth[wid] - 2];
                            if (lid == 0) compact_count[wid] = 0;
                            __syncwarp();

                            while (compact_count[wid] < WARP_SIZE && read_offset < total_read) {
                                uint32_t nbr = UINT32_MAX;
                                if (read_offset + lid < read_end) {
                                    nbr = index_gpu.nbrs_[pre_qe_idx][pre_dv]
                                          [read_offset + lid];
                                } else if (read_offset + lid < total_read) {
                                    nbr = update_index.nbrs_[pre_qe_idx][pre_dv]
                                          [read_offset + lid - read_end];
                                }

                                __syncwarp();
                                uint32_t ballot = __ballot_sync(0xffffffff, nbr != UINT32_MAX);
                                uint8_t num_new = __popc(ballot);
                                uint8_t my_rank = __popc(ballot & ((1u << lid) - 1u));

                                if (nbr != UINT32_MAX) {
                                    compact_nbrs[wid][compact_count[wid] + my_rank] = nbr;
                                    compact_pos[wid][compact_count[wid] + my_rank] = read_offset + lid;
                                }
                                if (lid == 0) compact_count[wid] += num_new;
                                read_offset += WARP_SIZE;
                                __syncwarp();
                            }

                            uint32_t num_valid = min(WARP_SIZE, (uint32_t)compact_count[wid]);
                            uint32_t temp_nbr = lid < num_valid
                                ? compact_nbrs[wid][lid] : UINT32_MAX;
                            uint32_t temp_pos = lid < num_valid
                                ? compact_pos[wid][lid] : UINT32_MAX;

                            if (num_valid == 0) {
                                if (lid == 0) {
                                    end_v[wid][dfs_depth[wid] - 2] += 1u;
                                    end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                                }
                                __syncwarp();
                            } else {
                                if (lid == num_valid - 1) {
                                    uint32_t total_check_end = min(temp_pos + 1, total_read);
                                    if (total_check_end >= total_read) {
                                        end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                                        end_v[wid][dfs_depth[wid] - 2] += 1u;
                                    } else {
                                        end_nbr[wid][dfs_depth[wid] - 2] = total_check_end;
                                    }
                                }
                                __syncwarp();
                            }

                            // Dedup check
                            bool found = lid < num_valid;
                            if (found) {
                                if (temp_nbr == v0 || temp_nbr == cur_v1)
                                    found = false;
                                for (uint8_t i = 2u; i < dfs_depth[wid] && found; i++) {
                                    if (result_queue[wid][i - 2u][queue_pos[wid][i - 2]] == temp_nbr)
                                        found = false;
                                }
                            }

                            // BN verification via lower_bound
                            if (found) {
                                for (uint8_t off = C_INDEXING_ORDERS[ei].bni_offs_[dfs_depth[wid]] + 1;
                                     off < C_INDEXING_ORDERS[ei].bni_offs_[dfs_depth[wid] + 1]; off++) {
                                    uint8_t bni = C_INDEXING_ORDERS[ei].bni_[off];
                                    uint8_t check_qe = C_EIDX[
                                        C_INDEXING_ORDERS[ei].vs_[bni] * C_QV_COUNT + current_qv];
                                    uint32_t check_v = (bni == 0) ? v0 :
                                        (bni == 1) ? cur_v1 :
                                        result_queue[wid][bni - 2][queue_pos[wid][bni - 2]];

                                    // Search temp_nbr in check_v's adjacency list (dual-source)
                                    uint32_t sz1 = index_gpu.sizes_[check_qe][check_v];
                                    bool uv3 = (C_DIR_TO_EDGE[check_qe] < ei);
                                    uint32_t sz2 = uv3 ? update_index.sizes_[check_qe][check_v] : 0u;

                                    bool f = false;
                                    if (sz1 > 0u) {
                                        uint32_t p = lower_bound(
                                            index_gpu.nbrs_[check_qe][check_v], sz1, temp_nbr);
                                        if (p < sz1 && index_gpu.nbrs_[check_qe][check_v][p] == temp_nbr)
                                            f = true;
                                    }
                                    if (!f && sz2 > 0u) {
                                        uint32_t p = lower_bound(
                                            update_index.nbrs_[check_qe][check_v], sz2, temp_nbr);
                                        if (p < sz2 && update_index.nbrs_[check_qe][check_v][p] == temp_nbr)
                                            f = true;
                                    }
                                    if (!f) { found = false; break; }
                                }
                            }
                            __syncwarp();

                            // Set support mask + update DFS queue
                            const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                            const uint32_t rank = __popc((UINT32_MAX >> (WARP_SIZE - lid)) & found_mask);

                            if (found) {
                                // Compute new path mask by ANDing parent path mask with candidate's support mask
                                // parent mask: path_masks[wid][0] for depth 2, path_masks[wid][d-2] for depth d >= 3
                                smask_t parent_pm = (dfs_depth[wid] == 2)
                                    ? path_masks[wid][0]
                                    : path_masks[wid][dfs_depth[wid] - 2];
                                // Note: at this stage (Phase 1.5), masks for depth 2+ are SMASK_ZERO
                                // (only u0/u1 were set by Phase 1). We use parent_pm directly as new_pm
                                // since the DFS is the FIRST to set masks at these depths.
                                smask_t new_pm = parent_pm;
                                if (new_pm != SMASK_ZERO) {
                                    atomicOr(&d_edge_sm_ptrs[ei * MAX_VCOUNT + current_qv][temp_nbr], new_pm);
                                    if (dfs_depth[wid] < end_depth - 1) {
                                        result_queue[wid][dfs_depth[wid] - 2][rank] = temp_nbr;
                                        // Store path mask for children at slot d-1
                                        if (rank == lid)
                                            path_masks[wid][dfs_depth[wid] - 1] = new_pm;
                                    }
                                } else {
                                    found = false;
                                }
                            }
                            __syncwarp();

                            // Recompute found_mask after support mask filtering
                            const uint32_t final_found_mask = __ballot_sync(0xffffffff, found);
                            const uint32_t final_rank = __popc((UINT32_MAX >> (WARP_SIZE - lid)) & final_found_mask);

                            if (found && final_rank == 0) {
                                queue_pos[wid][dfs_depth[wid] - 2] =
                                    (dfs_depth[wid] < end_depth - 1) ? 0u : __popc(final_found_mask);
                                queue_size[wid][dfs_depth[wid] - 2] = __popc(final_found_mask);
                            }
                            __syncwarp();
                        }
                    }
                    else
                    {
                        // Queue has candidates, go deeper
                        if (lid == 0 && dfs_depth[wid] < end_depth - 1) {
                            // Update path_masks for the current candidate before going deeper
                            uint8_t cur_d = dfs_depth[wid];
                            uint32_t vi = result_queue[wid][cur_d - 2][queue_pos[wid][cur_d - 2]];
                            uint8_t qi = C_INDEXING_ORDERS[ei].vs_[cur_d];
                            smask_t parent_pm = (cur_d == 2)
                                ? path_masks[wid][0]
                                : path_masks[wid][cur_d - 3];
                            path_masks[wid][cur_d - 1] = parent_pm
                                & d_edge_sm_ptrs[ei * MAX_VCOUNT + qi][vi];
                            dfs_depth[wid]++;
                        }
                        __syncwarp();
                    }
                } // while (dfs_depth >= 2)
            } // while (valid_mask for v1)
        } // for j (v1 enumeration)
    } // for v0
}

__global__ void writeInitialShallowDFS(
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
) {
    const uint8_t ei = blockIdx.y;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + warp_id;
    uint32_t num_warps = gridDim.x * blockDim.x / WARP_SIZE;

    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[0];
    uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[1];
    uint8_t idx01 = C_EIDX[u0 * C_QV_COUNT + u1];
    smask_t* sm_u0 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u0];
    smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1];

    __shared__ unsigned long long int write_pos[NWARP_PER_BLOCK];

    for (uint32_t v0 = gwarp_id; v0 < C_DV_COUNT; v0 += num_warps)
    {
        if (sm_u0[v0] == SMASK_ZERO) continue;
        uint32_t up_sz = update_index.sizes_[idx01][v0];

        // Count valid v1 per lane
        uint32_t my_count = 0;
        for (uint32_t j = lane_id; j < up_sz; j += WARP_SIZE) {
            uint32_t v1 = update_index.nbrs_[idx01][v0][j];
            if (v1 != UINT32_MAX && sm_u1[v1] != SMASK_ZERO)
                my_count++;
        }

        // Warp-level inclusive prefix sum
        uint32_t inclusive = my_count;
        for (uint32_t d = 1; d < WARP_SIZE; d *= 2) {
            uint32_t n = __shfl_up_sync(0xffffffff, inclusive, d);
            if (lane_id >= d) inclusive += n;
        }
        uint32_t lane_offset = inclusive - my_count;
        uint32_t total = __shfl_sync(0xffffffff, inclusive, WARP_SIZE - 1);
        if (lane_id == 0) write_pos[warp_id] = atomicAdd(new_res_size, (unsigned long long)total);
        __syncwarp();

        // Write [ei, v0, v1] entries for valid pairs
        uint32_t local_pos = lane_offset;
        for (uint32_t j = lane_id; j < up_sz; j += WARP_SIZE) {
            uint32_t v1 = update_index.nbrs_[idx01][v0][j];
            if (v1 != UINT32_MAX && sm_u1[v1] != SMASK_ZERO) {
                unsigned long long base = new_res + (write_pos[warp_id] + local_pos) * 3;
                C_RES_QUEUE.array_[base % C_RES_QUEUE.capability_] = (uint32_t)ei;
                C_RES_QUEUE.array_[(base + 1) % C_RES_QUEUE.capability_] = v0;
                C_RES_QUEUE.array_[(base + 2) % C_RES_QUEUE.capability_] = v1;
                local_pos++;
            }
        }
    }
}

__global__ void shallowDFSExpandFromQueue(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs,
    unsigned long long int res,
    const unsigned long long int res_size,
    const uint8_t end_depth
) {
    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    const unsigned long long gwarp_id = (unsigned long long)blockDim.x * blockIdx.x / WARP_SIZE + wid;
    if (gwarp_id >= res_size) return;

    // Read [ei, v0, v1] from C_RES_QUEUE
    unsigned long long base = (res + gwarp_id * 3) % C_RES_QUEUE.capability_;
    uint8_t ei = (uint8_t)C_RES_QUEUE.array_[base];
    uint32_t v0 = C_RES_QUEUE.array_[(base + 1) % C_RES_QUEUE.capability_];
    uint32_t v1 = C_RES_QUEUE.array_[(base + 2) % C_RES_QUEUE.capability_];

    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[0];
    uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[1];
    uint8_t idx01 = C_EIDX[u0 * C_QV_COUNT + u1];
    smask_t* sm_u0 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u0];
    smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1];
    (void)sm_u0; (void)sm_u1; (void)idx01;

    // Compute base path mask: AND of support masks for v0 and v1
    smask_t pm_v0 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u0][v0];
    smask_t pm_v1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1][v1];
    smask_t base_pm = pm_v0 & pm_v1;
    if (base_pm == SMASK_ZERO) return; // no common support bit

    // Shared memory DFS stack (same layout as shallowDFSExpand)
    __shared__ uint32_t result_queue[NWARP_PER_BLOCK][MAX_VCOUNT - 2][WARP_SIZE];
    __shared__ uint8_t queue_pos[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t queue_size[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t end_v[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint32_t end_nbr[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ bool intersection_continue[NWARP_PER_BLOCK][MAX_VCOUNT - 2];
    __shared__ uint8_t dfs_depth[NWARP_PER_BLOCK];
    __shared__ uint32_t compact_nbrs[NWARP_PER_BLOCK][WARP_SIZE * 2];
    __shared__ uint32_t compact_pos[NWARP_PER_BLOCK][WARP_SIZE * 2];
    __shared__ uint8_t compact_count[NWARP_PER_BLOCK];
    // Per-warp path masks for support mask tracking through DFS depths
    // path_masks[wid][0] = base mask (sm_u0[v0] & sm_u1[v1])
    // path_masks[wid][d-1] = base & sm[u2][v2] & ... & sm[u_{d}][v_{d}] for depth d >= 2
    __shared__ smask_t path_masks[NWARP_PER_BLOCK][MAX_VCOUNT - 2];

    // Initialize DFS state
    if (lid == 0) dfs_depth[wid] = 2;
    if (lid < MAX_VCOUNT - 2) {
        queue_pos[wid][lid] = 0u;
        queue_size[wid][lid] = 0u;
        end_v[wid][lid] = 0u;
        end_nbr[wid][lid] = 0u;
        intersection_continue[wid][lid] = false;
    }
    if (lid == 0) compact_count[wid] = 0;
    // Initialize path_masks: slot 0 = base mask (v0 & v1)
    if (lid < MAX_VCOUNT - 2) path_masks[wid][lid] = SMASK_ZERO;
    if (lid == 0) path_masks[wid][0] = base_pm;
    __syncwarp();

    // DFS main loop (identical to shallowDFSExpand)
    while (dfs_depth[wid] >= 2) {
        __syncwarp();

        const uint8_t& pre_qv_idx =
            C_INDEXING_ORDERS[ei].bni_[C_INDEXING_ORDERS[ei].bni_offs_[dfs_depth[wid]]];
        const uint8_t& pre_qe_idx =
            C_EIDX[C_INDEXING_ORDERS[ei].vs_[pre_qv_idx] * C_QV_COUNT +
                    C_INDEXING_ORDERS[ei].vs_[dfs_depth[wid]]];

        if (queue_pos[wid][dfs_depth[wid] - 2] >= queue_size[wid][dfs_depth[wid] - 2])
        {
            // Queue exhausted: backtrack or continue scanning
            if (intersection_continue[wid][dfs_depth[wid] - 2] &&
                ((pre_qv_idx < 2 && end_v[wid][dfs_depth[wid] - 2] > 0) ||
                 (pre_qv_idx >= 2 && end_v[wid][dfs_depth[wid] - 2] >
                  queue_pos[wid][pre_qv_idx - 2])))
            {
                if (lid == 0) {
                    queue_pos[wid][dfs_depth[wid] - 2] = 0u;
                    queue_size[wid][dfs_depth[wid] - 2] = 0u;
                    intersection_continue[wid][dfs_depth[wid] - 2] = false;
                    dfs_depth[wid]--;
                    if (dfs_depth[wid] >= 2)
                        queue_pos[wid][dfs_depth[wid] - 2]++;
                }
                __syncwarp();
            }
            else
            {
                // Scan new candidates
                if (!intersection_continue[wid][dfs_depth[wid] - 2]) {
                    if (lid == 0) {
                        if (pre_qv_idx < 2) {
                            end_v[wid][dfs_depth[wid] - 2] = 0u;
                            end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                            compact_count[wid] = 0;
                        } else {
                            end_v[wid][dfs_depth[wid] - 2] =
                                queue_pos[wid][pre_qv_idx - 2];
                            end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                            compact_count[wid] = 0;
                        }
                    }
                    __syncwarp();
                }
                intersection_continue[wid][dfs_depth[wid] - 2] = true;

                // Get predecessor data vertex
                uint32_t pre_dv = UINT32_MAX;
                if (pre_qv_idx == 0u) pre_dv = v0;
                else if (pre_qv_idx == 1u) pre_dv = v1;
                else pre_dv = result_queue[wid][pre_qv_idx - 2]
                                  [queue_pos[wid][pre_qv_idx - 2]];

                uint8_t current_qv = C_INDEXING_ORDERS[ei].vs_[dfs_depth[wid]];

                // Dual-source neighbor read + compaction
                uint32_t read_end = index_gpu.sizes_[pre_qe_idx][pre_dv];
                bool up_vis2 = (C_DIR_TO_EDGE[pre_qe_idx] < ei);
                uint32_t up_read_sz = up_vis2
                    ? update_index.sizes_[pre_qe_idx][pre_dv] : 0u;
                uint32_t total_read = read_end + up_read_sz;

                uint32_t read_offset = end_nbr[wid][dfs_depth[wid] - 2];
                if (lid == 0) compact_count[wid] = 0;
                __syncwarp();

                while (compact_count[wid] < WARP_SIZE && read_offset < total_read) {
                    uint32_t nbr = UINT32_MAX;
                    if (read_offset + lid < read_end) {
                        nbr = index_gpu.nbrs_[pre_qe_idx][pre_dv]
                              [read_offset + lid];
                    } else if (read_offset + lid < total_read) {
                        nbr = update_index.nbrs_[pre_qe_idx][pre_dv]
                              [read_offset + lid - read_end];
                    }

                    __syncwarp();
                    uint32_t ballot = __ballot_sync(0xffffffff, nbr != UINT32_MAX);
                    uint8_t num_new = __popc(ballot);
                    uint8_t my_rank = __popc(ballot & ((1u << lid) - 1u));

                    if (nbr != UINT32_MAX) {
                        compact_nbrs[wid][compact_count[wid] + my_rank] = nbr;
                        compact_pos[wid][compact_count[wid] + my_rank] = read_offset + lid;
                    }
                    if (lid == 0) compact_count[wid] += num_new;
                    read_offset += WARP_SIZE;
                    __syncwarp();
                }

                uint32_t num_valid = min(WARP_SIZE, (uint32_t)compact_count[wid]);
                uint32_t temp_nbr = lid < num_valid
                    ? compact_nbrs[wid][lid] : UINT32_MAX;
                uint32_t temp_pos = lid < num_valid
                    ? compact_pos[wid][lid] : UINT32_MAX;

                if (num_valid == 0) {
                    if (lid == 0) {
                        end_v[wid][dfs_depth[wid] - 2] += 1u;
                        end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                    }
                    __syncwarp();
                } else {
                    if (lid == num_valid - 1) {
                        uint32_t total_check_end = min(temp_pos + 1, total_read);
                        if (total_check_end >= total_read) {
                            end_nbr[wid][dfs_depth[wid] - 2] = 0u;
                            end_v[wid][dfs_depth[wid] - 2] += 1u;
                        } else {
                            end_nbr[wid][dfs_depth[wid] - 2] = total_check_end;
                        }
                    }
                    __syncwarp();
                }

                // Dedup check
                bool found = lid < num_valid;
                if (found) {
                    if (temp_nbr == v0 || temp_nbr == v1)
                        found = false;
                    for (uint8_t i = 2u; i < dfs_depth[wid] && found; i++) {
                        if (result_queue[wid][i - 2u][queue_pos[wid][i - 2]] == temp_nbr)
                            found = false;
                    }
                }

                // BN verification via lower_bound
                if (found) {
                    for (uint8_t off = C_INDEXING_ORDERS[ei].bni_offs_[dfs_depth[wid]] + 1;
                         off < C_INDEXING_ORDERS[ei].bni_offs_[dfs_depth[wid] + 1]; off++) {
                        uint8_t bni = C_INDEXING_ORDERS[ei].bni_[off];
                        uint8_t check_qe = C_EIDX[
                            C_INDEXING_ORDERS[ei].vs_[bni] * C_QV_COUNT + current_qv];
                        uint32_t check_v = (bni == 0) ? v0 :
                            (bni == 1) ? v1 :
                            result_queue[wid][bni - 2][queue_pos[wid][bni - 2]];

                        // Search temp_nbr in check_v's adjacency list (dual-source)
                        uint32_t sz1 = index_gpu.sizes_[check_qe][check_v];
                        bool uv3 = (C_DIR_TO_EDGE[check_qe] < ei);
                        uint32_t sz2 = uv3 ? update_index.sizes_[check_qe][check_v] : 0u;

                        bool f = false;
                        if (sz1 > 0u) {
                            uint32_t p = lower_bound(
                                index_gpu.nbrs_[check_qe][check_v], sz1, temp_nbr);
                            if (p < sz1 && index_gpu.nbrs_[check_qe][check_v][p] == temp_nbr)
                                f = true;
                        }
                        if (!f && sz2 > 0u) {
                            uint32_t p = lower_bound(
                                update_index.nbrs_[check_qe][check_v], sz2, temp_nbr);
                            if (p < sz2 && update_index.nbrs_[check_qe][check_v][p] == temp_nbr)
                                f = true;
                        }
                        if (!f) { found = false; break; }
                    }
                }
                __syncwarp();

                // Set support mask + update DFS queue
                const uint32_t found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t rank = __popc((UINT32_MAX >> (WARP_SIZE - lid)) & found_mask);

                if (found) {
                    // Compute new path mask by ANDing parent path mask with candidate's support mask
                    // parent mask: path_masks[wid][0] for depth 2, path_masks[wid][d-2] for depth d >= 3
                    smask_t parent_pm = (dfs_depth[wid] == 2)
                        ? path_masks[wid][0]
                        : path_masks[wid][dfs_depth[wid] - 2];
                    // Note: at this stage (Phase 1.5), masks for depth 2+ are SMASK_ZERO
                    // (only u0/u1 were set by Phase 1). We use parent_pm directly as new_pm
                    // since the DFS is the FIRST to set masks at these depths.
                    smask_t new_pm = parent_pm;
                    if (new_pm != SMASK_ZERO) {
                        atomicOr(&d_edge_sm_ptrs[ei * MAX_VCOUNT + current_qv][temp_nbr], new_pm);
                        if (dfs_depth[wid] < end_depth - 1) {
                            result_queue[wid][dfs_depth[wid] - 2][rank] = temp_nbr;
                            // Store path mask for children at slot d-1
                            if (rank == lid)
                                path_masks[wid][dfs_depth[wid] - 1] = new_pm;
                        }
                    } else {
                        found = false;
                    }
                }
                __syncwarp();

                // Recompute found_mask after support mask filtering
                const uint32_t final_found_mask = __ballot_sync(0xffffffff, found);
                const uint32_t final_rank = __popc((UINT32_MAX >> (WARP_SIZE - lid)) & final_found_mask);

                if (found && final_rank == 0) {
                    queue_pos[wid][dfs_depth[wid] - 2] =
                        (dfs_depth[wid] < end_depth - 1) ? 0u : __popc(final_found_mask);
                    queue_size[wid][dfs_depth[wid] - 2] = __popc(final_found_mask);
                }
                __syncwarp();
            }
        }
        else
        {
            // Queue has candidates, go deeper
            if (lid == 0 && dfs_depth[wid] < end_depth - 1) {
                // Update path_masks for the current candidate before going deeper
                uint8_t cur_d = dfs_depth[wid];
                uint32_t vi = result_queue[wid][cur_d - 2][queue_pos[wid][cur_d - 2]];
                uint8_t qi = C_INDEXING_ORDERS[ei].vs_[cur_d];
                smask_t parent_pm = (cur_d == 2)
                    ? path_masks[wid][0]
                    : path_masks[wid][cur_d - 3];
                path_masks[wid][cur_d - 1] = parent_pm
                    & d_edge_sm_ptrs[ei * MAX_VCOUNT + qi][vi];
                dfs_depth[wid]++;
            }
            __syncwarp();
        }
    } // while (dfs_depth >= 2)
}

__global__ void checkEdgeEmptyAll(
    const RelationsGPU update_index,
    uint8_t* d_flags
) {
    uint8_t ei = blockIdx.y;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nt = gridDim.x * blockDim.x;

    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[0];
    uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[1];
    uint8_t i01 = C_EIDX[u0 * C_QV_COUNT + u1];
    uint8_t i10 = C_EIDX[u1 * C_QV_COUNT + u0];

    bool has_01 = false, has_10 = false;
    for (uint32_t dv = tid; dv < C_DV_COUNT; dv += nt) {
        if (update_index.sizes_[i01][dv] > 0u) has_01 = true;
        if (update_index.sizes_[i10][dv] > 0u) has_10 = true;
    }
    // Write to two separate regions: [0..QE_COUNT-1] for dir01, [QE_COUNT..2*QE_COUNT-1] for dir10
    if (has_01) atomicOr(reinterpret_cast<uint32_t*>(d_flags) + (ei >> 2u), 1u << (8u * (ei & 3u)));
    if (has_10) atomicOr(reinterpret_cast<uint32_t*>(d_flags) + ((C_QE_COUNT + ei) >> 2u), 1u << (8u * ((C_QE_COUNT + ei) & 3u)));
}

// ============== 1-bit Packed Support Mask Kernels (SUPPORT_MASK_WIDTH == 1) ==============
// Storage: 32 vertices packed into one uint32_t. Each bit = one vertex active.
// Access:  SMASK_IDX(dv) = dv >> 5, SMASK_BIT(dv) = 1u << (dv & 31)

#if SUPPORT_MASK_WIDTH == 1

__global__ void setInitialValidBitsAll1Bit(
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nt = gridDim.x * blockDim.x;

    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[0];
    uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[1];
    uint8_t i01 = C_EIDX[u0 * C_QV_COUNT + u1];
    uint8_t i10 = C_EIDX[u1 * C_QV_COUNT + u0];

    for (uint32_t dv = tid; dv < C_DV_COUNT; dv += nt) {
        // For 1-bit: just check if vertex has any neighbor in update_index
        if (update_index.sizes_[i01][dv] > 0u)
            atomicOr(&d_edge_sm_ptrs[ei * MAX_VCOUNT + u0][SMASK_IDX(dv)], SMASK_BIT(dv));
        if (update_index.sizes_[i10][dv] > 0u)
            atomicOr(&d_edge_sm_ptrs[ei * MAX_VCOUNT + u1][SMASK_IDX(dv)], SMASK_BIT(dv));
    }
}

__global__ void checkAllConstraintsAndSetBitsAll1Bit(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
) {
    uint8_t ei = blockIdx.y;
    uint8_t u0 = C_INDEXING_ORDERS[ei].vs_[depth];

    // rebuild_config early exit: skip non-rebuild vertices (already SMASK_ALL)
    if (C_REBUILD_V_FLAGS[ei * MAX_VCOUNT + u0] == 0) return;

    uint32_t wid = threadIdx.x / WARP_SIZE;
    uint32_t lid = threadIdx.x % WARP_SIZE;
    uint32_t gwid = wid + (blockIdx.x * blockDim.x) / WARP_SIZE;
    uint32_t nwarp = (gridDim.x * blockDim.x) / WARP_SIZE;

    auto bn_s = C_INDEXING_ORDERS[ei].bni_offs_[depth];
    auto bn_e = C_INDEXING_ORDERS[ei].bni_offs_[depth + 1];
    uint8_t num_bn = bn_e - bn_s;

    uint32_t num_words = SMASK_DV_STRIDE(C_DV_COUNT);

    // Warp-per-word: each warp processes one word (32 vertices), each lane handles one vertex
    // 如何才能判断出来是否是被提前激活了呢？有两条边都能激活，需要带着模式边吗？还是怎么着呢？一起去做BFS算法，但是这条边也有可能分裂出来，呃，怎么办？？？冲突域检测冲突域检测冲突域检测
    for (uint32_t wi = gwid; wi < num_words; wi += nwarp) {
        uint32_t dv = wi * 32u + lid;
        bool is_valid = (dv < C_DV_COUNT);

        for (uint8_t bp = 0; bp < num_bn && is_valid; bp++) {
            uint8_t u1 = C_INDEXING_ORDERS[ei].vs_[C_INDEXING_ORDERS[ei].bni_[bn_s + bp]];
            uint8_t bwd = C_EIDX[u0 * C_QV_COUNT + u1];
            smask_t* sm_u1 = d_edge_sm_ptrs[ei * MAX_VCOUNT + u1];
            bool up_vis = (C_DIR_TO_EDGE[bwd] <= ei);

            bool bn_satisfied = false;

            // Check index_gpu neighbors for this vertex's backward direction
            if (index_gpu.sizes_[bwd][dv] > 0u) {
                for (uint32_t j = 0; j < index_gpu.sizes_[bwd][dv] && !bn_satisfied; j++) {
                    uint32_t nb = index_gpu.nbrs_[bwd][dv][j];
                    if (SMASK_TEST(sm_u1, nb)) bn_satisfied = true;
                }
            }

            // Check update_index neighbors if visible
            if (!bn_satisfied && up_vis && update_index.sizes_[bwd][dv] > 0u) {
                for (uint32_t j = 0; j < update_index.sizes_[bwd][dv] && !bn_satisfied; j++) {
                    uint32_t nb = update_index.nbrs_[bwd][dv][j];
                    if (SMASK_TEST(sm_u1, nb)) bn_satisfied = true;
                }
            }

            if (!bn_satisfied) is_valid = false;
        }

        // Combine all 32 lanes' results into one word via ballot
        uint32_t word_result = __ballot_sync(0xffffffff, is_valid);
        // Lane 0 writes directly (no atomicOr needed — each word processed by exactly one warp)
        if (lid == 0)
            d_edge_sm_ptrs[ei * MAX_VCOUNT + u0][wi] = word_result;
    }
}

#endif // SUPPORT_MASK_WIDTH == 1
