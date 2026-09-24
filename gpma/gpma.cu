#include "gpma.cuh"
#include <string>
#include <shared/timer.hpp>
#include <thrust/count.h>
#include <cuda_profiler_api.h>

#define cErr(errcode) { gpuAssert((errcode), __FILE__, __LINE__); }
__inline__ __host__ __device__
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
#if defined(__CUDA_ARCH__)
        // Device code: cannot call exit(), use assert to trap the thread
        assert(0);
#else
        exit(-1);
#endif
    }
}

__forceinline__ __host__ __device__
SIZE_TYPE fls(SIZE_TYPE x) {
    SIZE_TYPE r = 32;
    if (!x)
        return 0;
    if (!(x & 0xffff0000u))
        x <<= 16, r -= 16;
    if (!(x & 0xff000000u))
        x <<= 8, r -= 8;
    if (!(x & 0xf0000000u))
        x <<= 4, r -= 4;
    if (!(x & 0xc0000000u))
        x <<= 2, r -= 2;
    if (!(x & 0x80000000u))
        x <<= 1, r -= 1;
    return r;
}

template<typename T>
__global__
void memcpy_kernel(T *dest, const T *src, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        dest[i] = src[i];
    }
}

template<typename T>
__global__
void memset_kernel(T *data, T value, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        data[i] = value;
    }
}

template<typename KEY_TYPE_T, typename VALUE_TYPE_T>
__device__
void cub_sort_key_value_device(KEY_TYPE_T *keys, VALUE_TYPE_T *values, SIZE_TYPE size, KEY_TYPE_T *tmp_keys, VALUE_TYPE_T *tmp_values, cudaStream_t *stream) {
    cub::DoubleBuffer<KEY_TYPE_T> d_keys(keys, tmp_keys);
    cub::DoubleBuffer<VALUE_TYPE_T> d_values(values, tmp_values);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    // DeviceRadixSort 提供设备范围内的并行操作，用于计算驻留在设备可访问内存中的数据项序列的基数排序。
    // 对keys进行升序排序同时对values进行排序
    // https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
    // https://blog.csdn.net/hanqu3456/article/details/117950995
    cudaError_t err;
    err = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, size, 0, sizeof(KEY_TYPE_T) * 8, *stream);
    if (err != cudaSuccess) return;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) return;
    err = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, size, 0, sizeof(KEY_TYPE_T) * 8, *stream);
    if (err != cudaSuccess) { cudaFree(d_temp_storage); return; }
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    SIZE_TYPE THREADS_NUM = 128;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    memcpy_kernel<KEY_TYPE_T><<<BLOCKS_NUM, THREADS_NUM, 0, *stream>>>(d_keys.Alternate(), d_keys.Current(), size);
    memcpy_kernel<VALUE_TYPE_T><<<BLOCKS_NUM, THREADS_NUM, 0, *stream>>>(d_values.Alternate(), d_values.Current(), size);
}

template<typename KEY_TYPE_T, typename VALUE_TYPE_T>
__host__  
void cub_sort_key_value_host(KEY_TYPE_T *keys, VALUE_TYPE_T *values, SIZE_TYPE size, KEY_TYPE_T *tmp_keys, VALUE_TYPE_T *tmp_values) {
    cub::DoubleBuffer<KEY_TYPE_T> d_keys(keys, tmp_keys);
    cub::DoubleBuffer<VALUE_TYPE_T> d_values(values, tmp_values);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    // DeviceRadixSort 提供设备范围内的并行操作，用于计算驻留在设备可访问内存中的数据项序列的基数排序。
    // 对keys进行升序排序同时对values进行排序
    // https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
    // https://blog.csdn.net/hanqu3456/article/details/117950995
    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, size, 0, sizeof(KEY_TYPE_T) * 8));
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, size, 0, sizeof(KEY_TYPE_T) * 8));
    cErr(cudaDeviceSynchronize());
    cudaFree(d_temp_storage);

    SIZE_TYPE THREADS_NUM = 128;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    memcpy_kernel<KEY_TYPE_T><<<BLOCKS_NUM, THREADS_NUM>>>(d_keys.Alternate(), d_keys.Current(), size);
    memcpy_kernel<VALUE_TYPE_T><<<BLOCKS_NUM, THREADS_NUM>>>(d_values.Alternate(), d_values.Current(), size);
}

// 从 from = key >> 32 提取 block_id（高 5 位 = 边方向索引）
__device__ __host__ inline
SIZE_TYPE get_block_id_from(KEY_TYPE from) {
    return from >> 27;
}

// 从 from = key >> 32 提取顶点 ID（低 27 位）
__device__ __host__ inline
SIZE_TYPE get_vertex_id_from(KEY_TYPE from) {
    return from & 0x07FFFFFF;
}

__device__
KEY_TYPE handle_del_mod(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE seg_length, KEY_TYPE key,
        VALUE_TYPE value, KEY_TYPE leaf, SIZE_TYPE **outDegree, SIZE_TYPE *block_edge_num, SIZE_TYPE *update_add_size) {
    SIZE_TYPE value_last = VALUE_NONE;
    for (SIZE_TYPE i = 0; i < seg_length; i++) {
        if (keys[i] == key) {
            value_last = values[i];
            values[i] = value;
            leaf = KEY_NONE;
            break;
        }
    }
    if (VALUE_NONE == value && leaf != KEY_NONE) {
        KEY_TYPE from_raw = key >> 32;
        SIZE_TYPE from = get_vertex_id_from(from_raw);
        SIZE_TYPE block_id = get_block_id_from(from_raw);
        atomicAdd(&outDegree[block_id][from], 1);
        atomicAdd(&block_edge_num[block_id], 1);
        leaf = KEY_NONE;
    }
    if(VALUE_NONE != value && leaf == KEY_NONE && value_last != VALUE_NONE) {
        KEY_TYPE from_raw = key >> 32;
        SIZE_TYPE from = get_vertex_id_from(from_raw);
        SIZE_TYPE block_id = get_block_id_from(from_raw);
        atomicAdd(&outDegree[block_id][from], -1);
        atomicAdd(&block_edge_num[block_id], -1);
        atomicAdd(&update_add_size[block_id], -1);
    }
    return leaf;
}

__global__
void locate_leaf_kernel(KEY_TYPE **keyss, VALUE_TYPE **valuess, SIZE_TYPE *seg_lengthes,
        SIZE_TYPE *tree_heights, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
        KEY_TYPE *leaf, SIZE_TYPE **outDegree, SIZE_TYPE *block_edge_num, SIZE_TYPE *update_add_size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset) {
        KEY_TYPE key = update_keys[i];
        VALUE_TYPE value = update_values[i];
        KEY_TYPE from = (key >> 32);
        SIZE_TYPE block_id = get_block_id_from(from);
        KEY_TYPE *keys  = keyss[block_id];
        VALUE_TYPE *values = valuess[block_id];
        SIZE_TYPE seg_length = seg_lengthes[block_id];
        SIZE_TYPE tree_height = tree_heights[block_id];
        SIZE_TYPE seg_length_fls = fls(seg_length) - 1;
        SIZE_TYPE tree_size_fls = fls(seg_length << tree_height) - 1;
        KEY_TYPE prefix = 0;
        SIZE_TYPE current_bit = seg_length << tree_height >> 1;

        while (seg_length <= current_bit) {
            if (keys[prefix | current_bit] <= key)
                prefix |= current_bit;
            current_bit >>= 1;
        }

        prefix = handle_del_mod(keys + prefix, values + prefix, seg_length, key, value, prefix, outDegree, block_edge_num, update_add_size);
        if(prefix == KEY_NONE) {
            leaf[i] = KEY_NONE;
        } else {
            leaf[i] = (KEY_TYPE)((KEY_TYPE)(((KEY_TYPE)seg_length_fls) << 56) | (((KEY_TYPE)block_id) << 32) | prefix); // 前你八位代表seg_length, 剩余24位代表要更新的block_id，后32位代表位置
        }
    }
}

template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__
void block_compact_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE &compacted_size) {    
    typedef cub::BlockScan<SIZE_TYPE, THREAD_PER_BLOCK> BlockScan;
    SIZE_TYPE thread_id = threadIdx.x;

    KEY_TYPE *block_keys = keys;
    VALUE_TYPE *block_values = values;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];

    SIZE_TYPE thread_offset = thread_id * ITEM_PER_THREAD;
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_keys[i] = block_keys[thread_offset + i];
        thread_values[i] = block_values[thread_offset + i];
        block_keys[thread_offset + i] = KEY_NONE;
    }

    __shared__ typename BlockScan::TempStorage temp_storage;
    SIZE_TYPE thread_data[ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_data[i] = (thread_keys[i] == KEY_NONE || thread_values[i] == VALUE_NONE) ? 0 : 1;
    }
    __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
    __syncthreads();

    __shared__ SIZE_TYPE exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        exscan[i + thread_offset] = thread_data[i];
    }
    __syncthreads();

    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        if (thread_id == THREAD_PER_BLOCK - 1 && i == ITEM_PER_THREAD - 1)
            continue;
        if (exscan[thread_offset + i] != exscan[thread_offset + i + 1]) {
            SIZE_TYPE loc = exscan[thread_offset + i];
            block_keys[loc] = thread_keys[i];
            block_values[loc] = thread_values[i];
        }
    }

    // special logic for the last element
    if (thread_id == THREAD_PER_BLOCK - 1) {
        SIZE_TYPE loc = exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD - 1];
        if (thread_keys[ITEM_PER_THREAD - 1] == KEY_NONE || thread_values[ITEM_PER_THREAD - 1] == VALUE_NONE) {
            compacted_size = loc;
        } else {
            compacted_size = loc + 1;
            block_keys[loc] = thread_keys[ITEM_PER_THREAD - 1];
            block_values[loc] = thread_values[ITEM_PER_THREAD - 1];
        }
    }
}

template<typename FIRST_TYPE, typename SECOND_TYPE>
__device__
void block_pair_copy_kernel(FIRST_TYPE *dest_first, SECOND_TYPE *dest_second, FIRST_TYPE *src_first,
        SECOND_TYPE *src_second, SIZE_TYPE size) {
    for (SIZE_TYPE i = threadIdx.x; i < size; i += blockDim.x) {
        dest_first[i] = src_first[i];
        dest_second[i] = src_second[i];
    }
}

template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__
void block_redispatch_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE rebalance_width, SIZE_TYPE seg_length,
        SIZE_TYPE merge_size, SIZE_TYPE *row_offset, SIZE_TYPE update_node) {
    // step1: load KV in shared memory
    __shared__ KEY_TYPE block_keys[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    __shared__ VALUE_TYPE block_values[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    block_pair_copy_kernel<KEY_TYPE, VALUE_TYPE>(block_keys, block_values, keys, values, rebalance_width);
    __syncthreads();

    // step2: sort by key with value on shared memory
    typedef cub::BlockLoad<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockKeyLoadT;
    typedef cub::BlockLoad<VALUE_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockValueLoadT;
    typedef cub::BlockStore<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockKeyStoreT;
    typedef cub::BlockStore<VALUE_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockValueStoreT;
    typedef cub::BlockRadixSort<KEY_TYPE, THREAD_PER_BLOCK, ITEM_PER_THREAD, VALUE_TYPE> BlockRadixSortT;

    __shared__ union {
        typename BlockKeyLoadT::TempStorage key_load;
        typename BlockValueLoadT::TempStorage value_load;
        typename BlockKeyStoreT::TempStorage key_store;
        typename BlockValueStoreT::TempStorage value_store;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;
    // https://nvlabs.github.io/cub/classcub_1_1_block_load.html

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];
    BlockKeyLoadT(temp_storage.key_load).Load(block_keys, thread_keys);
    BlockValueLoadT(temp_storage.value_load).Load(block_values, thread_values);
    __syncthreads();

    BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values);
    __syncthreads();

    BlockKeyStoreT(temp_storage.key_store).Store(block_keys, thread_keys);
    BlockValueStoreT(temp_storage.value_store).Store(block_values, thread_values);
    __syncthreads();

    // step3: evenly re-dispatch KVs to leaf segments
    KEY_TYPE frac = rebalance_width / seg_length;
    KEY_TYPE deno = merge_size;
    for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x) {
        keys[i] = KEY_NONE;
    }
    __syncthreads();

    for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x) {
        SIZE_TYPE seg_idx = (SIZE_TYPE) (frac * i / deno);
        SIZE_TYPE seg_lane = (SIZE_TYPE) (frac * i % deno / frac);
        SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;

        KEY_TYPE cur_key = block_keys[i];
        VALUE_TYPE cur_value = block_values[i];
        keys[proj_location] = cur_key;
        values[proj_location] = cur_value;
        // addition for csr
        if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE) {
            SIZE_TYPE cur_row = (SIZE_TYPE) (cur_key >> 32) & 0x07FFFFFF;
            row_offset[cur_row + 1] = proj_location + update_node;
        }
    }
}

template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__global__
void block_rebalancing_kernel(SIZE_TYPE rebalance_width, SIZE_TYPE *seg_lengthes, KEY_TYPE **keyss, VALUE_TYPE **valuess,
        KEY_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, KEY_TYPE *unique_update_nodes,
        SIZE_TYPE *update_offset, SIZE_TYPE **lower_boundes, SIZE_TYPE **upper_boundes, SIZE_TYPE **row_offset, SIZE_TYPE *keys_sizes, KEY_TYPE *all_valid_update_size) {
    SIZE_TYPE update_id = blockIdx.x;
    SIZE_TYPE update_node = unique_update_nodes[update_id] & COL_IDX_NONE;
    KEY_TYPE block_id = (unique_update_nodes[update_id] >> 32) & 0xFF'FFFF;
    KEY_TYPE *keys = keyss[block_id];
    VALUE_TYPE *values = valuess[block_id];
    SIZE_TYPE seg_length = seg_lengthes[block_id];
    SIZE_TYPE level = fls(rebalance_width / seg_length) - 1;
    SIZE_TYPE lower_bound = lower_boundes[block_id][level];
    SIZE_TYPE upper_bound = upper_boundes[block_id][level];
    keys = keys + update_node;
    values = values + update_node;

    // compact
    __shared__ SIZE_TYPE compacted_size;
    block_compact_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(keys, values, compacted_size);
    __syncthreads();

    // judge whether fit the density threshold
    SIZE_TYPE interval_a = update_offset[update_id];
    SIZE_TYPE interval_b = update_offset[update_id + 1];
    SIZE_TYPE interval_size = interval_b - interval_a;
    SIZE_TYPE merge_size = compacted_size + interval_size;
    __syncthreads();
    assert(update_node + rebalance_width <= keys_sizes[block_id]);
    if (lower_bound <= merge_size && merge_size <= upper_bound) {
        block_pair_copy_kernel<KEY_TYPE, VALUE_TYPE>(keys + compacted_size, values + compacted_size,
                update_keys + interval_a, update_values + interval_a, interval_size);
        __syncthreads();

        // set KEY_NONE for executed update
        for (SIZE_TYPE i = interval_a + threadIdx.x; i < interval_b; i += blockDim.x) {
            update_nodes[i] = KEY_NONE;
        }

        // re-dispatch
        block_redispatch_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(keys, values, rebalance_width, seg_length,
                merge_size, row_offset[block_id], update_node);
    }
}

__global__
void label_key_whether_none_kernel(SIZE_TYPE *label, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        label[i] = (keys[i] == KEY_NONE || values[i] == VALUE_NONE) ? 0 : 1;
    }
}

__global__
void copy_compacted_kv(SIZE_TYPE *exscan, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_TYPE *tmp_keys,
        VALUE_TYPE *tmp_values, SIZE_TYPE *compacted_size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            tmp_keys[loc] = keys[i];
            tmp_values[loc] = values[i];
        }
    }
    
    if (0 == global_thread_id) {
        SIZE_TYPE loc = exscan[size - 1];
        if (keys[size - 1] == KEY_NONE || values[size - 1] == VALUE_NONE) {
            *compacted_size = loc;
        } else {
            *compacted_size = loc + 1;
            tmp_keys[loc] = keys[size - 1]; // 这个loc一定没有空隙
            tmp_values[loc] = values[size - 1];
        }
    }
}

__global__
void copy_compacted_kv_row_offset(SIZE_TYPE *exscan, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_TYPE *tmp_keys,
        VALUE_TYPE *tmp_values, SIZE_TYPE *row_offset) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            KEY_TYPE cur_key = keys[i];
            VALUE_TYPE cur_value = values[i];
            tmp_keys[loc] = cur_key;
            tmp_values[loc] = cur_value;
            // addition for csr
            if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE) {
                SIZE_TYPE cur_row = (SIZE_TYPE) (cur_key >> 32) & 0x07FFFFFF;
                row_offset[cur_row + 1] = loc;
            }
        }
    }
    
    if (0 == global_thread_id) {
        if (keys[size - 1] == KEY_NONE || values[size - 1] == VALUE_NONE) {
            ;
        } else {
            SIZE_TYPE loc = exscan[size - 1];
            KEY_TYPE cur_key = keys[size - 1];
            VALUE_TYPE cur_value = values[size - 1];
            tmp_keys[loc] = cur_key;
            tmp_values[loc] = cur_value;
            if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE) {
                SIZE_TYPE cur_row = (SIZE_TYPE) (cur_key >> 32) & 0x07FFFFFF;
                row_offset[cur_row + 1] = loc;
            }
        }
    }
}

__device__
void label_key_whether_none_kernel_after(SIZE_TYPE size, SIZE_TYPE *d_exscan, SIZE_TYPE *d_label, cudaStream_t *stream) {
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;
    cudaError_t err;
    err = cub::DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, d_label, d_exscan, size, *stream);
    if (err != cudaSuccess) return;
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) return;
    err = cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_label, d_exscan, size, *stream);
    if (err != cudaSuccess) { cudaFree(d_temp_storage); return; }
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage); 
}

__host__
void label_key_whether_none_kernel_after_host(SIZE_TYPE size, SIZE_TYPE *d_exscan, SIZE_TYPE *d_label) {
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;
    cErr(cub::DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, d_label, d_exscan, size));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    assert(d_temp_storage != NULL);
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_label, d_exscan, size));
    cudaFree(d_temp_storage); 
}

// 重新分发，每个顶点需要重新插入
__global__
void redispatch_kernel(KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, KEY_TYPE *keys, VALUE_TYPE *values,
        SIZE_TYPE update_width, SIZE_TYPE seg_length, SIZE_TYPE merge_size, SIZE_TYPE *row_offset,
        SIZE_TYPE update_node) { // update_width是更新后gpma.keys.size()
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    KEY_TYPE frac = update_width / seg_length; // 段数目
    KEY_TYPE deno = merge_size; // 元素个数

    for (SIZE_TYPE i = global_thread_id; i < merge_size; i += block_offset) {
        SIZE_TYPE seg_idx = (SIZE_TYPE) (frac * i / deno); // 每个段多少个元素 段数目 * 当前元素 / 总元素数目，判断出位于那个段
        SIZE_TYPE seg_lane = (SIZE_TYPE) (frac * i % deno / frac);// 段内位移，段数目 * 当前元素 % 总元素数目 / 段数目 取余之后两个间隔一定大于frac，这个不会出现冲突
        SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;
        KEY_TYPE cur_key = tmp_keys[i];
        VALUE_TYPE cur_value = tmp_values[i];
        keys[proj_location] = cur_key;
        values[proj_location] = cur_value;
        // addition for csr
        if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE) { // 判断是否是卫兵
            SIZE_TYPE cur_row = (SIZE_TYPE) (cur_key >> 32) & 0x07FFFFFF;
            row_offset[cur_row + 1] = proj_location + update_node;
        }
    }
}

__global__
void rebalancing_kernel(SIZE_TYPE unique_update_size, SIZE_TYPE update_width, SIZE_TYPE *seg_lengthes, KEY_TYPE **keyss,
        VALUE_TYPE **valuess, KEY_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values,
        KEY_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE **lower_boundes, SIZE_TYPE **upper_boundes,
        SIZE_TYPE **row_offset, SIZE_TYPE *keys_sizes, KEY_TYPE *all_valid_update_size) {
    cudaStream_t stream = 0;

    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    KEY_TYPE *tmp_keys = nullptr;
    VALUE_TYPE *tmp_values = nullptr;
    SIZE_TYPE *tmp_exscan = nullptr, *tmp_label = nullptr, *compacted_size = nullptr;

    // Allocate temp buffers with explicit error checking (no exit on failure)
    cudaError_t err;
    err = cudaMalloc(&compacted_size, sizeof(SIZE_TYPE));
    if (err != cudaSuccess) return;
    err = cudaMalloc(&tmp_keys, update_width * sizeof(KEY_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); return; }
    err = cudaMalloc(&tmp_values, update_width * sizeof(VALUE_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); cudaFree(tmp_keys); return; }
    err = cudaMalloc(&tmp_exscan, update_width * sizeof(SIZE_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); cudaFree(tmp_keys); cudaFree(tmp_values); return; }
    err = cudaMalloc(&tmp_label, update_width * sizeof(SIZE_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); cudaFree(tmp_keys); cudaFree(tmp_values); cudaFree(tmp_exscan); return; }

    for (SIZE_TYPE i = global_thread_id; i < unique_update_size; i += block_offset) {
        SIZE_TYPE block_id = (unique_update_nodes[i] >> 32) & 0xFF'FFFF;
        SIZE_TYPE seg_length = seg_lengthes[block_id];
        SIZE_TYPE level = fls(update_width / seg_length) - 1;
        SIZE_TYPE tree_height = fls(keys_sizes[block_id] / seg_length) - 1;
        if(level != tree_height / 3 - 1 && level != tree_height)
            continue;
        SIZE_TYPE update_node = (unique_update_nodes[i] & COL_IDX_NONE);
        assert(update_node + update_width <= keys_sizes[block_id]);
        KEY_TYPE *keys = keyss[block_id];
        VALUE_TYPE *values = valuess[block_id];
        SIZE_TYPE lower_bound = lower_boundes[block_id][level];
        SIZE_TYPE upper_bound = upper_boundes[block_id][level];

        keys = keys + update_node;
        values = values + update_node;
        SIZE_TYPE interval_a = update_offset[i];
        SIZE_TYPE interval_b = update_offset[i + 1];

        SIZE_TYPE THREADS_NUM = 32;
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
        label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_label, keys, values, update_width);
        label_key_whether_none_kernel_after(update_width, tmp_exscan, tmp_label, &stream);
        // copy compacted kv to tmp, and set the original to none
        copy_compacted_kv<<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_exscan, keys, values, update_width, tmp_keys, tmp_values, compacted_size);
        cudaDeviceSynchronize();

        // judge whether fit the density threshold
        SIZE_TYPE interval_size = interval_b - interval_a;
        SIZE_TYPE merge_size = (*compacted_size) + interval_size; // compacted_size interval_size

        if (lower_bound <= merge_size && merge_size <= upper_bound) {
            // move
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, interval_size);
            memcpy_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_keys + (*compacted_size),
                    update_keys + interval_a, interval_size);
            memcpy_kernel<VALUE_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_values + (*compacted_size),
                    update_values + interval_a, interval_size);
            // set KEY_NONE for executed updates
            memset_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(update_nodes + interval_a, KEY_NONE, interval_size);

            cub_sort_key_value_device<KEY_TYPE, VALUE_TYPE>(tmp_keys, tmp_values, merge_size, keys, values, &stream);
            // re-dispatch
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
            memset_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(keys, KEY_NONE, update_width);
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
            redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_keys, tmp_values, keys, values, update_width, seg_length,
                    merge_size, row_offset[block_id], update_node);
        }
        cudaDeviceSynchronize();
    }
    cudaFree(compacted_size);
    cudaFree(tmp_keys);
    cudaFree(tmp_values);
    cudaFree(tmp_exscan);
    cudaFree(tmp_label);
}

// 这个函数主要实现的功能为混合对多个PMA数组的 tree_height / 3 层次和 tree_height 层次进行更新，来尝试解决自底向上层层尝试时间过长的问题
// 在第一次尝试中尝试对所有的tree_height / 3和tree_height混合进行更新，之后再次提高高度尝试进行更新，tree_height高度
__global__
void rebalancing_kernel_hybird(SIZE_TYPE unique_update_size, SIZE_TYPE *seg_lengthes, KEY_TYPE **keyss,
        VALUE_TYPE **valuess, KEY_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values,
        KEY_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE **lower_boundes, SIZE_TYPE **upper_boundes,
        SIZE_TYPE **row_offset, SIZE_TYPE *keys_sizes, KEY_TYPE *all_valid_update_size) {
    cudaStream_t stream = 0;
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    KEY_TYPE *tmp_keys = nullptr;
    VALUE_TYPE *tmp_values = nullptr;
    SIZE_TYPE *tmp_exscan = nullptr, *tmp_label = nullptr, *compacted_size = nullptr;
    SIZE_TYPE max_update_width = 0;
    for (SIZE_TYPE i = global_thread_id; i < unique_update_size; i += block_offset) {
        SIZE_TYPE update_width = (1 << (unique_update_nodes[i] >> 56));
        max_update_width = max(max_update_width, update_width);
    }

    // Allocate temp buffers with explicit error checking (no exit on failure)
    cudaError_t err;
    err = cudaMalloc(&compacted_size, sizeof(SIZE_TYPE));
    if (err != cudaSuccess) return;
    err = cudaMalloc(&tmp_keys, max_update_width * sizeof(KEY_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); return; }
    err = cudaMalloc(&tmp_values, max_update_width * sizeof(VALUE_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); cudaFree(tmp_keys); return; }
    err = cudaMalloc(&tmp_exscan, max_update_width * sizeof(SIZE_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); cudaFree(tmp_keys); cudaFree(tmp_values); return; }
    err = cudaMalloc(&tmp_label, max_update_width * sizeof(SIZE_TYPE));
    if (err != cudaSuccess) { cudaFree(compacted_size); cudaFree(tmp_keys); cudaFree(tmp_values); cudaFree(tmp_exscan); return; }

    for (SIZE_TYPE i = global_thread_id; i < unique_update_size; i += block_offset) {
        SIZE_TYPE update_width = (1 << (unique_update_nodes[i] >> 56));
        SIZE_TYPE block_id = (unique_update_nodes[i] >> 32) & 0xFF'FFFF;
        SIZE_TYPE seg_length = seg_lengthes[block_id];
        SIZE_TYPE level = fls(update_width / seg_length) - 1;
        SIZE_TYPE tree_height = fls(keys_sizes[block_id] / seg_length) - 1;
        if(level != tree_height / 3 - 1 && level != tree_height)
            continue;
        SIZE_TYPE update_node = (unique_update_nodes[i] & COL_IDX_NONE);
        assert(update_node + update_width <= keys_sizes[block_id]);
        KEY_TYPE *keys = keyss[block_id];
        VALUE_TYPE *values = valuess[block_id];
        SIZE_TYPE lower_bound = lower_boundes[block_id][level];
        SIZE_TYPE upper_bound = upper_boundes[block_id][level];

        keys = keys + update_node;
        values = values + update_node;
        SIZE_TYPE interval_a = update_offset[i];
        SIZE_TYPE interval_b = update_offset[i + 1];

        SIZE_TYPE THREADS_NUM = 32;
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
        label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_label, keys, values, update_width);
        label_key_whether_none_kernel_after(update_width, tmp_exscan, tmp_label, &stream);
        // copy compacted kv to tmp, and set the original to none
        copy_compacted_kv<<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_exscan, keys, values, update_width, tmp_keys, tmp_values, compacted_size);
        cudaDeviceSynchronize();

        // judge whether fit the density threshold
        SIZE_TYPE interval_size = interval_b - interval_a;
        SIZE_TYPE merge_size = (*compacted_size) + interval_size; // compacted_size interval_size
        if (lower_bound <= merge_size && merge_size <= upper_bound) {
            // move
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, interval_size);
            memcpy_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_keys + (*compacted_size),
                    update_keys + interval_a, interval_size);
            memcpy_kernel<VALUE_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_values + (*compacted_size),
                    update_values + interval_a, interval_size);
            // set KEY_NONE for executed updates
            memset_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(update_nodes + interval_a, KEY_NONE, interval_size);

            cub_sort_key_value_device<KEY_TYPE, VALUE_TYPE>(tmp_keys, tmp_values, merge_size, keys, values, &stream);
            // re-dispatch
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
            memset_kernel<KEY_TYPE> <<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(keys, KEY_NONE, update_width);
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
            redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM, 0, stream>>>(tmp_keys, tmp_values, keys, values, update_width, seg_length,
                    merge_size, row_offset[block_id], update_node);
        }
        cudaDeviceSynchronize();
    }
    cudaFree(compacted_size);
    cudaFree(tmp_keys);
    cudaFree(tmp_values);
    cudaFree(tmp_exscan);
    cudaFree(tmp_label);
}

__device__
void recalculate_density_multi(SIZE_TYPE *seg_lengthes, SIZE_TYPE *tree_heights, SIZE_TYPE **lower_boundes, SIZE_TYPE **upper_boundes, SIZE_TYPE index, double density_upper_thres_leaf, double density_upper_thres_root, double density_lower_thres_root, double density_lower_thres_leaf) {
    SIZE_TYPE level_length = seg_lengthes[index];
    for (SIZE_TYPE i = 0; i <= tree_heights[index]; i++) {
        double density_lower = density_lower_thres_root
                + (density_lower_thres_leaf - density_lower_thres_root) * (tree_heights[index] - i)
                        / tree_heights[index];
        double density_upper = density_upper_thres_root
                + (density_upper_thres_leaf - density_upper_thres_root) * (tree_heights[index] - i)
                        / tree_heights[index];

        lower_boundes[index][i] = (SIZE_TYPE) ceil(density_lower * level_length);
        upper_boundes[index][i] = (SIZE_TYPE) floor(density_upper * level_length);

        // special trim for wrong threshold introduced by float-integer conversion
        if (0 < i) {
            lower_boundes[index][i] = max(lower_boundes[index][i], 2 * lower_boundes[index][i - 1]);
            upper_boundes[index][i] = min(upper_boundes[index][i], 2 * upper_boundes[index][i - 1]);
        }
        level_length <<= 1;
    }
}

__host__
void locate_leaf_batch(KEY_TYPE **keyss, VALUE_TYPE **valuess, SIZE_TYPE *seg_lengthes,
        SIZE_TYPE *tree_heights, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size,
        KEY_TYPE *leaf, SIZE_TYPE **outDegree, SIZE_TYPE *block_edge_num, SIZE_TYPE *update_add_size) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size); // 对更新进行分块并行
    locate_leaf_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(keyss, valuess, seg_lengthes, tree_heights, update_keys,
            update_values, update_size, leaf, outDegree, block_edge_num, update_add_size);
}

__host__
void rebalance_batch(SIZE_TYPE update_width, SIZE_TYPE* seg_lengthes, KEY_TYPE **keyss, VALUE_TYPE **valuess,
        KEY_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values,
        KEY_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE **lower_boundes,
        SIZE_TYPE **upper_boundes, SIZE_TYPE **row_offset, SIZE_TYPE unique_update_size, SIZE_TYPE *keys_sizes, KEY_TYPE *all_block_update_size, KEY_TYPE *all_block_valid_update_size, KEY_TYPE *all_kernel_update_size, KEY_TYPE *all_kernel_valid_update_size, float *rebalance_block_batch_time, float *rebalance_kernel_batch_time, float *single_rebalance_block_batch_time, float *single_rebalance_kernel_batch_time, KEY_TYPE *single_block_update_size, KEY_TYPE *single_kernel_update_size, SIZE_TYPE *all_block_update_time, SIZE_TYPE *all_kernel_update_time) {
    cErr(cudaDeviceSynchronize());
    Timer time1;
    time1.Start();
    if (update_width <= 1024) {
        (*all_block_update_time) += unique_update_size;
        (*all_block_update_size) += unique_update_size * update_width;
        (*single_block_update_size) += unique_update_size * update_width;
        // func pointer for each template
        void (*func_arr[10])(SIZE_TYPE, SIZE_TYPE *, KEY_TYPE**, VALUE_TYPE**, KEY_TYPE*, KEY_TYPE*, VALUE_TYPE*,
                KEY_TYPE*, SIZE_TYPE*, SIZE_TYPE**, SIZE_TYPE**, SIZE_TYPE**, SIZE_TYPE*, KEY_TYPE *);
        func_arr[0] = block_rebalancing_kernel<2, 1>; // Thread / block; Item / Thread
        func_arr[1] = block_rebalancing_kernel<4, 1>; 
        func_arr[2] = block_rebalancing_kernel<8, 1>;
        func_arr[3] = block_rebalancing_kernel<16, 1>;
        func_arr[4] = block_rebalancing_kernel<32, 1>;
        func_arr[5] = block_rebalancing_kernel<32, 2>;
        func_arr[6] = block_rebalancing_kernel<32, 4>;
        func_arr[7] = block_rebalancing_kernel<32, 8>;
        func_arr[8] = block_rebalancing_kernel<32, 16>;
        func_arr[9] = block_rebalancing_kernel<32, 32>;

        // operate each tree node by cuda-block
        SIZE_TYPE THREADS_NUM = update_width > 32 ? 32 : update_width;
        SIZE_TYPE BLOCKS_NUM = unique_update_size;
        func_arr[fls(update_width) - 2]<<<BLOCKS_NUM, THREADS_NUM>>>(update_width, seg_lengthes, keyss, valuess, update_nodes,
                update_keys, update_values, unique_update_nodes, update_offset, lower_boundes, upper_boundes, row_offset, keys_sizes, all_block_valid_update_size);
        cErr(cudaDeviceSynchronize());
        (*rebalance_block_batch_time) += time1.Finish();
        (*single_rebalance_block_batch_time) += time1.Finish();
    } else {
        (*all_kernel_update_time) += unique_update_size;
        (*all_kernel_update_size) += unique_update_size * update_width;
        (*single_kernel_update_size) += unique_update_size * update_width;

        // operate each tree node by cub-kernel (dynamic parallelsim)
        SIZE_TYPE BLOCKS_NUM = min(16, unique_update_size);

        rebalancing_kernel<<<BLOCKS_NUM, 1>>>(unique_update_size, update_width, seg_lengthes, keyss, valuess, update_nodes,
                update_keys, update_values, unique_update_nodes, update_offset, lower_boundes, upper_boundes, row_offset, keys_sizes, all_kernel_valid_update_size);
        cErr(cudaDeviceSynchronize());
        (*rebalance_kernel_batch_time) += time1.Finish();
        (*single_rebalance_kernel_batch_time) += time1.Finish();
    }
}

__global__
void memset_kernel_num(SIZE_TYPE *dest, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        dest[i] = i;
    }
}

template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__global__ 
void block_compact_insertions(KEY_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *update_size, SIZE_TYPE update_width, SIZE_TYPE *update_size_update_width) {
    SIZE_TYPE update_size_new = *update_size;
    typedef cub::BlockScan<SIZE_TYPE, THREAD_PER_BLOCK> BlockScanT;
    typedef cub::BlockReduce<SIZE_TYPE, THREAD_PER_BLOCK> BlockReduceT;
    __shared__ union {
        typename BlockScanT::TempStorage scan;
        typename BlockReduceT::TempStorage reduce;
    } temp_storage;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];
    KEY_TYPE thread_nodes[ITEM_PER_THREAD];
    SIZE_TYPE thread_labels[ITEM_PER_THREAD];
    SIZE_TYPE global_thread_id = threadIdx.x;
    for(SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_nodes[i] = KEY_NONE;
    }
    for(SIZE_TYPE i = 0; i < ITEM_PER_THREAD && (global_thread_id * ITEM_PER_THREAD + i) < update_size_new; i++) {
        SIZE_TYPE index = (global_thread_id * ITEM_PER_THREAD) + i;
        thread_keys[i] = update_keys[index];
        thread_values[i] = update_values[index];
        thread_nodes[i] = update_nodes[index];
        thread_labels[i] = (thread_nodes[i] == KEY_NONE) ? 0 : 1;
    }
    
    __syncthreads();
    BlockScanT(temp_storage.scan).ExclusiveSum(thread_labels, thread_labels);
    SIZE_TYPE count_block = 0;
    SIZE_TYPE count_block_update_width = 0;
    SIZE_TYPE update_width_fls = fls(update_width) - 1;
    for(int i = 0; i < ITEM_PER_THREAD && (global_thread_id * ITEM_PER_THREAD + i) < update_size_new; i++) {
        if(thread_nodes[i] != KEY_NONE) {
            update_keys[thread_labels[i]] = thread_keys[i];
            update_values[thread_labels[i]] = thread_values[i];
            update_nodes[thread_labels[i]] = thread_nodes[i];
            count_block++;
        }
        SIZE_TYPE update_width_fls_now = (thread_nodes[i] >> 56);
        count_block_update_width += (update_width_fls == update_width_fls_now);
    }
    __syncthreads();
    SIZE_TYPE block_sum = BlockReduceT(temp_storage.reduce).Sum(count_block);
    __syncthreads();
    SIZE_TYPE block_sum_update_width = BlockReduceT(temp_storage.reduce).Sum(count_block_update_width);
    if (threadIdx.x == 0) {
        (*update_size) = block_sum;
        (*update_size_update_width) = block_sum_update_width;
    }
}

__global__
void gather(KEY_TYPE *keys, VALUE_TYPE *values, KEY_TYPE *update_nodes, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, KEY_TYPE *tmp_update_nodes, KEY_TYPE *map, SIZE_TYPE keys_size, SIZE_TYPE *update_size, SIZE_TYPE update_width, SIZE_TYPE *update_size_update_width) {
    using BlockReduce = cub::BlockReduce<SIZE_TYPE, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage_block;
    __shared__ typename BlockReduce::TempStorage temp_storage_update_width;
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    SIZE_TYPE count_block = 0;
    SIZE_TYPE count_update_width = 0;
    SIZE_TYPE update_width_fls = fls(update_width) - 1;
    for (SIZE_TYPE i = global_thread_id; i < keys_size; i += block_offset) {
        if(update_nodes[i] != KEY_NONE) {
            tmp_keys[map[i]] = keys[i];
            tmp_values[map[i]] = values[i];
            tmp_update_nodes[map[i]] = update_nodes[i];
            count_block++;
        }
        SIZE_TYPE update_width_fls_now = (update_nodes[i] >> 56);
        count_update_width += (update_width_fls == update_width_fls_now);
    }
    __syncthreads();
    SIZE_TYPE block_sum = BlockReduce(temp_storage_block).Sum(count_block);
    SIZE_TYPE block_sum_update_width = BlockReduce(temp_storage_update_width).Sum(count_update_width);
    __syncthreads();
    if (threadIdx.x == 0) {
        update_size[blockIdx.x] = block_sum;
        update_size_update_width[blockIdx.x] = block_sum_update_width;
    }
}

__global__
void memcpy_count_if(KEY_TYPE *keys, VALUE_TYPE *values, KEY_TYPE *update_nodes, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, KEY_TYPE *tmp_update_nodes, SIZE_TYPE keys_size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < keys_size; i += block_offset) {
        keys[i] = tmp_keys[i];
        values[i] = tmp_values[i];
        update_nodes[i] = tmp_update_nodes[i];
    }
}

__global__
void label_update_nodes_whether_none_kernel(KEY_TYPE *label, KEY_TYPE *update_nodes, SIZE_TYPE size) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        label[i] = (update_nodes[i] == KEY_NONE) ? 0 : 1;
    }
}

__host__
void label_update_nodes_whether_none_kernel_after(SIZE_TYPE size, KEY_TYPE *d_exscan, KEY_TYPE *d_label) {
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;
    cErr(cub::DeviceScan::ExclusiveSum(NULL, temp_storage_bytes, d_label, d_exscan, size));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    assert(d_temp_storage != NULL);
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_label, d_exscan, size));
    cudaFree(d_temp_storage); 
}

__host__
void compact_insertions(KEY_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_keys_size, 
    SIZE_TYPE *update_size, SIZE_TYPE update_width, SIZE_TYPE *update_size_update_width, KEY_TYPE *tmp_keys_array, VALUE_TYPE *tmp_values_array, KEY_TYPE *tmp_label_array, KEY_TYPE *tmp_exscan_array) {
    SIZE_TYPE update_keys_size_new = *update_size;
    assert(update_keys_size_new <= update_keys_size);
    SIZE_TYPE THREADS_NUM = 256;
    SIZE_TYPE BLOCKS_NUM;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_keys_size_new);
    label_update_nodes_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_label_array, update_nodes, update_keys_size_new);
    label_update_nodes_whether_none_kernel_after(update_keys_size_new, tmp_exscan_array, tmp_label_array);
    (*update_size) = 0;
    (*update_size_update_width) = 0;
    SIZE_TYPE THREADS_NUM_1 = 256, BLOCKS_NUM_1;
    BLOCKS_NUM_1 = CALC_BLOCKS_NUM(THREADS_NUM_1, update_keys_size_new);

    SIZE_TYPE *d_update_size;
    SIZE_TYPE *d_update_size_update_width;
    cudaMalloc(&d_update_size, BLOCKS_NUM_1 * sizeof(SIZE_TYPE));
    cudaMalloc(&d_update_size_update_width, BLOCKS_NUM_1 * sizeof(SIZE_TYPE));

    gather<<<BLOCKS_NUM_1, THREADS_NUM_1>>> (update_keys, update_values, update_nodes, tmp_keys_array, tmp_values_array, tmp_label_array, tmp_exscan_array, update_keys_size_new, d_update_size, update_width, d_update_size_update_width);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_update_size, update_size, BLOCKS_NUM_1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_update_size, update_size, BLOCKS_NUM_1);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_update_size_update_width, update_size_update_width, BLOCKS_NUM_1);

    memcpy_count_if<<<BLOCKS_NUM, THREADS_NUM>>> (update_keys, update_values, update_nodes, tmp_keys_array, tmp_values_array, tmp_label_array, update_keys_size_new);

    cudaFree(d_update_size);
    cudaFree(d_update_size_update_width);
    cudaFree(d_temp_storage);
}

__host__
void compact_insertions_kernel(KEY_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_keys_size, 
    SIZE_TYPE *update_size, KEY_TYPE *tmp_keys_array, VALUE_TYPE *tmp_values_array, KEY_TYPE *tmp_label_array, KEY_TYPE *tmp_exscan_array, SIZE_TYPE update_width, SIZE_TYPE *update_size_update_width) {
    SIZE_TYPE update_size_now = *update_size;
    if (update_size_now <= 1024) {
        // func pointer for each template
        SIZE_TYPE thread_num[10] = {2, 4, 8, 16, 32, 32, 32, 32, 32, 32};
        void (*func_arr[10])(KEY_TYPE *, KEY_TYPE *, VALUE_TYPE *, SIZE_TYPE *, SIZE_TYPE, SIZE_TYPE *);
        func_arr[0] = block_compact_insertions<2, 1>; // Thread / block; Item / Thread
        func_arr[1] = block_compact_insertions<4, 1>;
        func_arr[2] = block_compact_insertions<8, 1>;
        func_arr[3] = block_compact_insertions<16, 1>;
        func_arr[4] = block_compact_insertions<32, 1>;
        func_arr[5] = block_compact_insertions<32, 2>;
        func_arr[6] = block_compact_insertions<32, 4>;
        func_arr[7] = block_compact_insertions<32, 8>;
        func_arr[8] = block_compact_insertions<32, 16>;
        func_arr[9] = block_compact_insertions<32, 32>;

        // operate each tree node by cuda-block
        // SIZE_TYPE THREADS_NUM = (*update_size) > 32 ? 32 : (*update_size);
        SIZE_TYPE index;
        if(update_size_now & (update_size_now - 1) == 0)
            index = min(9, fls((update_size_now)) - 2);
        else
            index = min(9, fls((update_size_now)) - 1);
        func_arr[index]<<<1, thread_num[index]>>>(update_nodes, update_keys, update_values, update_size, update_width, update_size_update_width);
    } else {
        compact_insertions(update_nodes, update_keys, update_values, update_keys_size, 
        update_size, update_width, update_size_update_width, tmp_keys_array, tmp_values_array, tmp_label_array, tmp_exscan_array);
    }
}

template<SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__global__
void block_set_update_offset(SIZE_TYPE *update_offset, SIZE_TYPE unique_node_size, SIZE_TYPE update_size, SIZE_TYPE *tmp_offset) {
    // 求取前缀和放到update_offset中
    SIZE_TYPE thread_tmp_offset[ITEM_PER_THREAD];
    using BlockScan = cub::BlockScan<SIZE_TYPE, THREAD_PER_BLOCK>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    SIZE_TYPE global_thread_id = threadIdx.x;
    for(SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_tmp_offset[i] = 0;
    }
    for(SIZE_TYPE i = 0; i < ITEM_PER_THREAD && (global_thread_id * ITEM_PER_THREAD) + i < unique_node_size; i++) {
        SIZE_TYPE index = (global_thread_id * ITEM_PER_THREAD) + i;
        thread_tmp_offset[i] = tmp_offset[index];
    }

    __syncthreads();
    SIZE_TYPE block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(thread_tmp_offset, thread_tmp_offset, block_aggregate);

    for(SIZE_TYPE i = 0; i < ITEM_PER_THREAD && (global_thread_id * ITEM_PER_THREAD) + i < unique_node_size; i++) {
        SIZE_TYPE index = (global_thread_id * ITEM_PER_THREAD) + i;
        update_offset[index] = thread_tmp_offset[i];
    }

    if(global_thread_id == 0) {
        *(update_offset + unique_node_size) = update_size;
    }
}

__host__
void set_update_offset(SIZE_TYPE *update_offset, SIZE_TYPE unique_node_size, SIZE_TYPE update_size, SIZE_TYPE *tmp_offset) {
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp_offset,
            update_offset, unique_node_size));
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp_offset,
            update_offset, unique_node_size));
    cudaFree(d_temp_storage);
    // update_offset      <-- [1, 3, 4, 7, 8, 8] update_offset
    memset_kernel<<<1, 1>>>(update_offset + unique_node_size, update_size, 1);
}

__host__
void set_update_offset_kernel(SIZE_TYPE *update_offset, SIZE_TYPE unique_node_size, SIZE_TYPE update_size, SIZE_TYPE *tmp_offset) {
    if (unique_node_size <= 1024) {
        // func pointer for each template
        SIZE_TYPE thread_num[10] = {2, 4, 8, 16, 32, 32, 32, 32, 32, 32};
        void (*func_arr[10])(SIZE_TYPE *, SIZE_TYPE, SIZE_TYPE , SIZE_TYPE *);
        func_arr[0] = block_set_update_offset<2, 1>; // Thread / block; Item / Thread
        func_arr[1] = block_set_update_offset<4, 1>;
        func_arr[2] = block_set_update_offset<8, 1>;
        func_arr[3] = block_set_update_offset<16, 1>;
        func_arr[4] = block_set_update_offset<32, 1>;
        func_arr[5] = block_set_update_offset<32, 2>;
        func_arr[6] = block_set_update_offset<32, 4>;
        func_arr[7] = block_set_update_offset<32, 8>;
        func_arr[8] = block_set_update_offset<32, 16>;
        func_arr[9] = block_set_update_offset<32, 32>;

        // operate each tree node by cuda-block
        SIZE_TYPE index;
        if(unique_node_size & (unique_node_size - 1) == 0)
            index = min(9, fls((unique_node_size)) - 2);
        else
            index = min(9, fls((unique_node_size)) - 1);
        func_arr[index]<<<1, thread_num[index]>>>(update_offset, unique_node_size, update_size, tmp_offset);
    } else {
        set_update_offset(update_offset, unique_node_size, update_size, tmp_offset);
    }
}

// 注意tmp_offset长度为update_size * SIZE_TYPE 
__host__
void compress_insertions_by_node(KEY_TYPE *update_nodes, SIZE_TYPE update_size,
        KEY_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE *unique_node_size, 
        SIZE_TYPE *tmp_offset) {
    // step1: encode
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    // DeviceRunLengthEncode 提供设备范围的并行操作，用于划分驻留在设备可访问内存中的序列中相同值项的“运行”。
    // https://nvlabs.github.io/cub/structcub_1_1_device_run_length_encode.html#ab25e5e8289fe198b8fea68ac5f010118
    // Determine temporary device storage requirements
    cErr(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, update_nodes,
        unique_update_nodes, tmp_offset, unique_node_size, update_size));
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run encoding
    cErr(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, update_nodes,
        unique_update_nodes, tmp_offset, unique_node_size, update_size));
    // d_unique_out      <-- [0, 2, 9, 5, 8] unique_update_nodes
    // d_counts_out      <-- [1, 2, 1, 3, 1] tmp_offset
    // d_num_runs_out    <-- [5] num_runs_out 不同的数量
    // step2: exclusive scan
    // DeviceScan 提供设备范围的并行操作，用于计算驻留在设备可访问内存中的一系列数据项的前缀扫描。
    cudaFree(d_temp_storage);
}

__global__
void up_level_kernel(KEY_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    SIZE_TYPE update_width_fls = fls(update_width) - 1;

    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset) {
        KEY_TYPE node = update_nodes[i];
        if(node != KEY_NONE) {
            KEY_TYPE update_width_key = (update_width >> 1);
            node = node & ~update_width_key;
            SIZE_TYPE update_width_fls_next = (node >> 56);
            if(update_width_fls_next == (update_width_fls - 1))
                update_width_fls_next++;
            node = (node & 0xFF'FFFF'FFFF'FFFF) | ((KEY_TYPE)update_width_fls_next << 56);
            update_nodes[i] = node;
        }
    }
}

__host__
void up_level_batch(KEY_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);
    up_level_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes, update_size, update_width);
}

__global__
void up_level_kernel_hybird_3(KEY_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE *seg_lengthes, SIZE_TYPE *keys_sizes) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset) {
        KEY_TYPE node = update_nodes[i];
        if(node != KEY_NONE) {
            SIZE_TYPE block_id = (node >> 32) & 0xFF'FFFF;
            KEY_TYPE seg_length = seg_lengthes[block_id];
            KEY_TYPE tree_height = fls(keys_sizes[block_id] / seg_length) - 1;
            KEY_TYPE tree_height_3 = tree_height / 3;
            KEY_TYPE update_width_next = seg_length << tree_height_3;
            if(update_width_next <= 1024)
                update_width_next = keys_sizes[block_id];
            SIZE_TYPE update_width_last = (node >> 56);
            SIZE_TYPE update_width_fls_next = fls(update_width_next) - 1;
            if(update_width_fls_next <= update_width_last)
                continue;
            node = node & (~(update_width_next - 1));
            node = (node & 0xFF'FFFF'FFFF'FFFF) | ((KEY_TYPE)update_width_fls_next << 56);
            update_nodes[i] = node;
        }
    }
}

__host__
void up_level_batch_hybird_3(KEY_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE *seg_lengthes, SIZE_TYPE *keys_sizes) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);
    up_level_kernel_hybird_3<<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes, update_size, seg_lengthes, keys_sizes);
}

__global__
void up_level_kernel_hybird(KEY_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE *seg_lengthes, SIZE_TYPE *keys_sizes) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = global_thread_id; i < update_size; i += block_offset) {
        KEY_TYPE node = update_nodes[i];
        if(node != KEY_NONE) {
            SIZE_TYPE block_id = (node >> 32) & 0xFF'FFFF;
            KEY_TYPE update_width_next = keys_sizes[block_id];
            node = node & (~(update_width_next - 1));
            SIZE_TYPE update_width_fls_next = fls(update_width_next) - 1;
            node = (node & 0xFF'FFFF'FFFF'FFFF) | ((KEY_TYPE)update_width_fls_next << 56);
            update_nodes[i] = node;
        }
    }
}

__host__
void up_level_batch_hybird(KEY_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE *seg_lengthes, SIZE_TYPE *keys_sizes) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);
    up_level_kernel_hybird<<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes, update_size, seg_lengthes, keys_sizes);
}

__global__
void resize_gpmas(SIZE_TYPE *keys_sizes, SIZE_TYPE *block_edge_num, SIZE_TYPE *seg_lengthes, SIZE_TYPE *tree_heights, SIZE_TYPE *keys_sizes_new, SIZE_TYPE **lower_boundes, SIZE_TYPE **upper_boundes, double density_upper_thres_leaf, double density_upper_thres_root, double density_lower_thres_root, double density_lower_thres_leaf, SIZE_TYPE num_blocks) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < num_blocks; i += block_offset) {
        SIZE_TYPE merge_size = block_edge_num[i];
        SIZE_TYPE original_tree_size = keys_sizes[i];
        SIZE_TYPE tree_size = 4;
        while (floor(density_upper_thres_root * tree_size) < merge_size)
            tree_size <<= 1;
        seg_lengthes[i] = 1 << (fls(fls(tree_size)) - 1);
        tree_heights[i] = fls(tree_size / seg_lengthes[i]) - 1;
        keys_sizes_new[i] = tree_size;
        recalculate_density_multi(seg_lengthes, tree_heights, lower_boundes, upper_boundes, i, density_upper_thres_leaf, density_upper_thres_root, density_lower_thres_root, density_lower_thres_leaf);

        if(upper_boundes[i][tree_heights[i]] < merge_size) {
            tree_size <<= 1;
            seg_lengthes[i] = 1 << (fls(fls(tree_size)) - 1);
            tree_heights[i] = fls(tree_size / seg_lengthes[i]) - 1;
            keys_sizes_new[i] = tree_size;
            recalculate_density_multi(seg_lengthes, tree_heights, lower_boundes, upper_boundes, i, density_upper_thres_leaf, density_upper_thres_root, density_lower_thres_root, density_lower_thres_leaf);
        }
        assert(merge_size >= lower_boundes[i][tree_heights[i]]);
        assert(merge_size <= upper_boundes[i][tree_heights[i]]);
    }
}

__host__
void update_gpma_stage3(Multi_GPMA *gpma) {
    // step3: extract insertions
    gpma->update_size = gpma->update_keys_size;
    gpma->unique_node_size = 0;
    if(gpma->update_width > 1024)
        return;
    compact_insertions_kernel(gpma->update_nodes, gpma->update_keys, gpma->update_values, gpma->update_keys_size,
    &(gpma->update_size), gpma->tmp_keys_array, gpma->tmp_values_array, gpma->tmp_label_array, gpma->tmp_exscan_array, gpma->update_width, &(gpma->update_size_update_width));
    cErr(cudaDeviceSynchronize());
    // step5: rebalance each tree level
    while(gpma->update_size > 0) {
        compress_insertions_by_node(gpma->update_nodes, gpma->update_size_update_width, gpma->unique_update_nodes, gpma->update_offset, &(gpma->unique_node_size), (SIZE_TYPE *)gpma->tmp_label_array);
        // printf("update_width %d update_size %d update_sizie_update_width %d unique_node_size %d\n", gpma->update_width, gpma->update_size, gpma->update_size_update_width, gpma->unique_node_size);
        cErr(cudaDeviceSynchronize());
        if(gpma->unique_node_size != 0) {
            set_update_offset_kernel(gpma->update_offset, gpma->unique_node_size, gpma->update_size_update_width, (SIZE_TYPE *)gpma->tmp_label_array);
            rebalance_batch(gpma->update_width, gpma->d_seg_lengthes, gpma->d_keyss, gpma->d_valuess, gpma->update_nodes,
            gpma->update_keys, gpma->update_values, gpma->unique_update_nodes,
            gpma->update_offset, gpma->d_lower_boundes, gpma->d_upper_boundes, gpma->row_offset, gpma->unique_node_size, gpma->d_keys_sizes, &gpma->all_block_update_size, &gpma->all_block_valid_update_size, &gpma->all_kernel_update_size, &gpma->all_kernel_valid_update_size, &gpma->rebalance_block_batch_time, &gpma->rebalance_kernel_batch_time, &gpma->single_rebalance_block_batch_time, &gpma->single_rebalance_kernel_batch_time, &gpma->single_block_update_size, &gpma->single_kernel_update_size, &gpma->all_block_update_time, &gpma->all_kernel_update_time);
        }
        gpma->update_width <<= 1;
        gpma->level ++;
        if(gpma->update_width > 1024)
            return;
        up_level_batch(gpma->update_nodes, gpma->update_size, gpma->update_width);
        compact_insertions_kernel(gpma->update_nodes, gpma->update_keys, gpma->update_values, gpma->update_keys_size, &(gpma->update_size),
        gpma->tmp_keys_array, gpma->tmp_values_array, gpma->tmp_label_array, gpma->tmp_exscan_array, gpma->update_width, &(gpma->update_size_update_width));
        cErr(cudaDeviceSynchronize());
    }
}

__host__
void update_gpma_stage4(Multi_GPMA *gpma) {
    // 进行1/3 tree_height高度的更新
    up_level_batch_hybird_3(gpma->update_nodes, gpma->update_size, gpma->d_seg_lengthes, gpma->d_keys_sizes);
    compact_insertions_kernel(gpma->update_nodes, gpma->update_keys, gpma->update_values, gpma->update_keys_size, &(gpma->update_size),
    gpma->tmp_keys_array, gpma->tmp_values_array, gpma->tmp_label_array, gpma->tmp_exscan_array, gpma->update_width, &(gpma->update_size_update_width));
    cErr(cudaDeviceSynchronize());

    if(gpma->update_size == 0)
        return ;
    
    compress_insertions_by_node(gpma->update_nodes, gpma->update_size, gpma->unique_update_nodes, gpma->update_offset, &(gpma->unique_node_size), (SIZE_TYPE *)gpma->tmp_label_array);
    cErr(cudaDeviceSynchronize());
    
    if(gpma->unique_node_size != 0) {
        set_update_offset_kernel(gpma->update_offset, gpma->unique_node_size, gpma->update_size, (SIZE_TYPE *)gpma->tmp_label_array);
        gpma->all_kernel_update_time += gpma->unique_node_size;
        gpma->single_kernel_update_size += gpma->unique_node_size;
        Timer time1;
        time1.Start();
        SIZE_TYPE BLOCKS_NUM = min(16, gpma->unique_node_size);
        rebalancing_kernel_hybird<<<BLOCKS_NUM, 1>>>(gpma->unique_node_size, gpma->d_seg_lengthes, gpma->d_keyss, gpma->d_valuess, gpma->update_nodes,
                gpma->update_keys, gpma->update_values, gpma->unique_update_nodes, gpma->update_offset, gpma->d_lower_boundes, gpma->d_upper_boundes, gpma->row_offset, gpma->d_keys_sizes, &(gpma->all_kernel_valid_update_size));
        cErr(cudaDeviceSynchronize());
        gpma->rebalance_kernel_batch_time += time1.Finish();
        gpma->single_rebalance_kernel_batch_time += time1.Finish();
    }

    // 进行tree_height高度的更新
    up_level_batch_hybird(gpma->update_nodes, gpma->update_size, gpma->d_seg_lengthes, gpma->d_keys_sizes);
    compact_insertions_kernel(gpma->update_nodes, gpma->update_keys, gpma->update_values, gpma->update_keys_size, &(gpma->update_size),
    gpma->tmp_keys_array, gpma->tmp_values_array, gpma->tmp_label_array, gpma->tmp_exscan_array, gpma->update_width, &(gpma->update_size_update_width));
    cErr(cudaDeviceSynchronize());
    if(gpma->update_size == 0)
        return ;

    compress_insertions_by_node(gpma->update_nodes, gpma->update_size, gpma->unique_update_nodes, gpma->update_offset, &(gpma->unique_node_size), (SIZE_TYPE *)gpma->tmp_label_array);
    cErr(cudaDeviceSynchronize());
    
    if(gpma->unique_node_size != 0) {
        set_update_offset_kernel(gpma->update_offset, gpma->unique_node_size, gpma->update_size, (SIZE_TYPE *)gpma->tmp_label_array);
        gpma->all_kernel_update_time += gpma->unique_node_size;
        gpma->single_kernel_update_size += gpma->unique_node_size;
        Timer time1;
        time1.Start();
        SIZE_TYPE BLOCKS_NUM = min(16, gpma->unique_node_size);
        rebalancing_kernel_hybird<<<BLOCKS_NUM, 1>>>(gpma->unique_node_size, gpma->d_seg_lengthes, gpma->d_keyss, gpma->d_valuess, gpma->update_nodes,
                gpma->update_keys, gpma->update_values, gpma->unique_update_nodes, gpma->update_offset, gpma->d_lower_boundes, gpma->d_upper_boundes, gpma->row_offset, gpma->d_keys_sizes, &(gpma->all_kernel_valid_update_size));
        cErr(cudaDeviceSynchronize());
        gpma->rebalance_kernel_batch_time += time1.Finish();
        gpma->single_rebalance_kernel_batch_time += time1.Finish();
    }
}

__global__
void gather_update_nodes(KEY_TYPE *keys, VALUE_TYPE *values, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *map, SIZE_TYPE keys_size) {
    using BlockReduce = cub::BlockReduce<SIZE_TYPE, 512>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < keys_size; i += block_offset) {
        tmp_keys[i] = keys[map[i]];
        tmp_values[i] = values[map[i]];
    }
}

__host__
void update_gpma_stage1(Multi_GPMA *gpma) {
    locate_leaf_batch(gpma->d_keyss, gpma->d_valuess,
    gpma->d_seg_lengthes, gpma->d_tree_heights,
            gpma->update_keys, gpma->update_values, gpma->update_keys_size, gpma->update_nodes, gpma->outDegree, gpma->block_edge_num, gpma->update_add_size);
    cErr(cudaDeviceSynchronize());
}

__global__
void change_update_level(KEY_TYPE *update_nodes, bool *need_sign_insert, SIZE_TYPE update_keys_size, SIZE_TYPE *keys_sizes) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = global_thread_id; i < update_keys_size; i += block_offset) {
        KEY_TYPE node = update_nodes[i];
        if(node != KEY_NONE) {
            KEY_TYPE block_id = (node >> 32) & 0xFF'FFFF; // 获取block_id，去除前5位
            if(need_sign_insert[block_id]) {
                SIZE_TYPE tree_size_fls = fls(keys_sizes[block_id]) - 1;
                node = (node & 0xFF'FFFF'0000'0000) | ((KEY_TYPE)tree_size_fls << 56);
                update_nodes[i] = node;
            }
            // KEY_TYPE update_width = (1 << (node >> 56));
            // if(update_width < 32 && keys_sizes[block_id] >= 32) {
            //     node = (node & (~(31ll)));
            //     node = (node & 0xFF'FFFF'FFFF'FFFF) | (5ll << 56);
            //     update_nodes[i] = node;
            // }
        }
    }
}

__host__
void update_gpma_stage2(Multi_GPMA *gpma) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, gpma->update_keys_size);
    change_update_level<<<BLOCKS_NUM, THREADS_NUM>>>(gpma->update_nodes, gpma->d_need_sign_insert, gpma->update_keys_size, gpma->d_keys_sizes);
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, gpma->update_keys_size);
    memset_kernel_num<<<BLOCKS_NUM, THREADS_NUM>>>((SIZE_TYPE *)gpma->tmp_label_array, gpma->update_keys_size);
    cub_sort_key_value_host<KEY_TYPE, SIZE_TYPE>(gpma->update_nodes, (SIZE_TYPE *)gpma->tmp_label_array, gpma->update_keys_size, gpma->tmp_keys_array, (SIZE_TYPE *)gpma->tmp_exscan_array);
    gather_update_nodes<<<BLOCKS_NUM, THREADS_NUM>>>(gpma->update_keys, gpma->update_values, gpma->tmp_keys_array, gpma->tmp_values_array, (SIZE_TYPE *)gpma->tmp_label_array, gpma->update_keys_size);
    memcpy_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(gpma->update_keys, gpma->tmp_keys_array, gpma->update_keys_size);
    memcpy_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(gpma->update_values, gpma->tmp_values_array, gpma->update_keys_size);
    cErr(cudaDeviceSynchronize());
}

__global__
void init_gpmas_keys_values(Multi_GPMA *gpma) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < gpma->num_blocks; i += block_offset) {
        gpma->d_keyss[i][0] = gpma->d_keyss[i][2] = KEY_MAX;
        gpma->d_keyss[i][1] = gpma->d_keyss[i][3] = KEY_NONE;
        gpma->d_valuess[i][0] = gpma->d_valuess[i][2] = 1;
        gpma->d_valuess[i][1] = gpma->d_valuess[i][3] = 0;
    }
}

__host__
void init_gpmas_keys_values_batch(Multi_GPMA *gpma) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, gpma->num_blocks);
    init_gpmas_keys_values<<<BLOCKS_NUM, THREADS_NUM>>>(gpma);
    cErr(cudaDeviceSynchronize());
}

__host__
void malloc_multi_gpma(Multi_GPMA *gpma) {
    gpma->update_width = 0xFFFF'FFFF; 
    Timer time1;
    time1.Start();
    SIZE_TYPE malloc_time = 0;
    cErr(cudaMemcpy(gpma->h_keys_sizes, gpma->d_keys_sizes, gpma->num_blocks * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
    cErr(cudaMemcpy(gpma->h_keys_sizes_new, gpma->d_keys_sizes_new, gpma->num_blocks * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
    cErr(cudaMemcpy(gpma->h_seg_lengthes, gpma->d_seg_lengthes, gpma->num_blocks * sizeof(SIZE_TYPE), cudaMemcpyDeviceToHost));
    for(SIZE_TYPE i = 0; i < gpma->num_blocks; i++) {
        if(gpma->h_keys_sizes[i] != gpma->h_keys_sizes_new[i] && gpma->h_keys_sizes_new[i] > gpma->h_keys_sizes[i]) {
            malloc_time++;
            KEY_TYPE *keys_new;
            VALUE_TYPE *values_new;
            SIZE_TYPE THREADS_NUM = 32;
            SIZE_TYPE BLOCKS_NUM;
            cErr(cudaMalloc(&keys_new, gpma->h_keys_sizes_new[i] * sizeof(KEY_TYPE))); // 这里cudaMalloc太多太碎了
            cErr(cudaMalloc(&values_new, gpma->h_keys_sizes_new[i] * sizeof(VALUE_TYPE)));
            if(gpma->keyss[i]) {
                BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, gpma->h_keys_sizes[i]);
                memcpy_kernel<KEY_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(keys_new, gpma->keyss[i], gpma->h_keys_sizes[i]);
                memcpy_kernel<VALUE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(values_new, gpma->valuess[i], gpma->h_keys_sizes[i]);
                cudaFree(gpma->keyss[i]);
                cudaFree(gpma->valuess[i]);
            }
            SIZE_TYPE memset_size = gpma->h_keys_sizes_new[i] - gpma->h_keys_sizes[i];
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, memset_size);
            memset_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(keys_new + gpma->h_keys_sizes[i], KEY_NONE, memset_size);
            memset_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(values_new + gpma->h_keys_sizes[i], VALUE_NONE, memset_size);
            gpma->keyss[i] = keys_new;  
            gpma->valuess[i] = values_new;
            gpma->h_keys_sizes[i] = gpma->h_keys_sizes_new[i];
            gpma->h_need_sign_insert[i] = true;
            gpma->update_width = min(gpma->update_width, gpma->h_keys_sizes[i]);
        } else if(gpma->h_keys_sizes[i] != gpma->h_keys_sizes_new[i] && gpma->h_keys_sizes_new[i] < gpma->h_keys_sizes[i]) {
            malloc_time++;
            SIZE_TYPE update_width = gpma->h_keys_sizes[i];
            SIZE_TYPE memset_size = gpma->h_keys_sizes_new[i];
            SIZE_TYPE *tmp_label, *tmp_exscan, BLOCKS_NUM, THREADS_NUM = 32;
            KEY_TYPE *keys_new;
            VALUE_TYPE *values_new;
            cErr(cudaMalloc(&tmp_label, update_width * sizeof(SIZE_TYPE)));
            cErr(cudaMalloc(&tmp_exscan, update_width * sizeof(SIZE_TYPE)));
            cErr(cudaMalloc(&keys_new, gpma->h_keys_sizes_new[i] * sizeof(KEY_TYPE)));
            cErr(cudaMalloc(&values_new, gpma->h_keys_sizes_new[i] * sizeof(VALUE_TYPE)));
            
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, memset_size);
            memset_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(keys_new, KEY_NONE, memset_size);
            memset_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(values_new, VALUE_NONE, memset_size);
           
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
            label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_label, gpma->keyss[i], gpma->valuess[i], update_width);
            label_key_whether_none_kernel_after_host(update_width, tmp_exscan, tmp_label);
            copy_compacted_kv_row_offset<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_exscan, gpma->keyss[i], gpma->valuess[i], update_width, keys_new, values_new, gpma->row_offset[i]);
            cudaDeviceSynchronize();
            cudaFree(tmp_label);
            cudaFree(tmp_exscan);
            cudaFree(gpma->keyss[i]);
            cudaFree(gpma->valuess[i]);
            gpma->keyss[i] = keys_new;  
            gpma->valuess[i] = values_new;
            gpma->h_keys_sizes[i] = gpma->h_keys_sizes_new[i];
            gpma->h_need_sign_insert[i] = true;
            gpma->update_width = min(gpma->update_width, gpma->h_keys_sizes[i]);
        } else {
            gpma->update_width = min(gpma->update_width, gpma->h_seg_lengthes[i]);
            gpma->h_need_sign_insert[i] = false;
        }
    }
    cErr(cudaMemcpy(gpma->d_keys_sizes, gpma->h_keys_sizes, gpma->num_blocks * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(gpma->d_need_sign_insert, gpma->h_need_sign_insert, gpma->num_blocks * sizeof(bool), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(gpma->d_keyss, gpma->keyss, gpma->num_blocks * sizeof(KEY_TYPE*), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(gpma->d_valuess, gpma->valuess, gpma->num_blocks * sizeof(VALUE_TYPE*), cudaMemcpyHostToDevice));
}

__host__
void resize_gpmas_batch(Multi_GPMA *gpma) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM;
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, gpma->num_blocks);
    resize_gpmas<<<BLOCKS_NUM, THREADS_NUM>>>((gpma->d_keys_sizes), gpma->block_edge_num, gpma->d_seg_lengthes, gpma->d_tree_heights, gpma->d_keys_sizes_new, gpma->d_lower_boundes, gpma->d_upper_boundes, gpma->density_upper_thres_leaf, gpma->density_upper_thres_root, gpma->density_lower_thres_root, gpma->density_lower_thres_leaf, gpma->num_blocks);
    cErr(cudaDeviceSynchronize());
    malloc_multi_gpma(gpma);
    cErr(cudaDeviceSynchronize());
}

__global__
void init_row_wall(KEY_TYPE *data, SIZE_TYPE size, SIZE_TYPE num_blocks) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < size; i += block_offset) {
        for(SIZE_TYPE j = 0; j < num_blocks; j++) {
            KEY_TYPE x = j << 27 | i;
            data[i * num_blocks + j] = (x << 32) + COL_IDX_NONE;
        }
    }
}

__global__
void init_row_wall_offset(KEY_TYPE *data, SIZE_TYPE start, SIZE_TYPE count, SIZE_TYPE num_blocks) {
    SIZE_TYPE global_thread_id = blockDim.x * blockIdx.x;
    SIZE_TYPE block_offset = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = global_thread_id; i < count; i += block_offset) {
        SIZE_TYPE vid = start + i;
        for(SIZE_TYPE j = 0; j < num_blocks; j++) {
            KEY_TYPE x = j << 27 | vid;
            data[i * num_blocks + j] = (x << 32) + COL_IDX_NONE;
        }
    }
}

Multi_GPMA::Multi_GPMA(SIZE_TYPE row_num_, SIZE_TYPE num_blocks_) {
    row_num = row_num_;
    update_size = 0;
    unique_node_size = 0;
    compacted_size = 0;
    update_size_update_width = 0;
    update_width = 0;
	all_block_update_size = 0;
	all_block_valid_update_size = 0;
    all_kernel_update_size = 0;
    all_kernel_valid_update_size = 0;
    rebalance_block_batch_time = 0;
    rebalance_kernel_batch_time = 0;

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM;
    num_blocks = num_blocks_;

    // Allocate 2D row_offset: row_offset[block_id][row_num + 1]
    SIZE_TYPE **h_row_offset;
    cErr(cudaMallocHost(&h_row_offset, num_blocks * sizeof(SIZE_TYPE*)));
    for (SIZE_TYPE i = 0; i < num_blocks; i++) {
        cErr(cudaMalloc(&h_row_offset[i], sizeof(SIZE_TYPE) * (row_num + 1)));
    }
    cErr(cudaMalloc(&row_offset, num_blocks * sizeof(SIZE_TYPE*)));
    cErr(cudaMemcpy(row_offset, h_row_offset, num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyHostToDevice));
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, (row_num + 1));
    for (SIZE_TYPE i = 0; i < num_blocks; i++) {
        memset_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(h_row_offset[i], 0, row_num + 1);
    }
    cudaFreeHost(h_row_offset);

    // Allocate 2D outDegree: outDegree[block_id][from]
    SIZE_TYPE **h_outDegree;
    cErr(cudaMallocHost(&h_outDegree, num_blocks * sizeof(SIZE_TYPE*)));
    for (SIZE_TYPE i = 0; i < num_blocks; i++) {
        cErr(cudaMalloc(&h_outDegree[i], sizeof(SIZE_TYPE) * row_num));
    }
    cErr(cudaMalloc(&outDegree, num_blocks * sizeof(SIZE_TYPE*)));
    cErr(cudaMemcpy(outDegree, h_outDegree, num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyHostToDevice));
    BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, row_num);
    for (SIZE_TYPE i = 0; i < num_blocks; i++) {
        memset_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(h_outDegree[i], 0, row_num);
    }
    cudaFreeHost(h_outDegree);
    cErr(cudaMallocHost(&keyss, num_blocks * sizeof(KEY_TYPE*)));
    cErr(cudaMallocHost(&valuess, num_blocks * sizeof(VALUE_TYPE*)));
    cErr(cudaMallocHost(&lower_boundes, num_blocks * sizeof(SIZE_TYPE*)));
    cErr(cudaMallocHost(&upper_boundes, num_blocks * sizeof(VALUE_TYPE*)));

    cErr(cudaMalloc(&d_keyss, num_blocks * sizeof(KEY_TYPE*)));
    cErr(cudaMalloc(&d_valuess, num_blocks * sizeof(VALUE_TYPE*)));
    cErr(cudaMalloc(&d_lower_boundes, num_blocks * sizeof(SIZE_TYPE*)));
    cErr(cudaMalloc(&d_upper_boundes, num_blocks * sizeof(VALUE_TYPE*)));

    cErr(cudaMalloc(&d_tree_heights, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMallocHost(&h_seg_lengthes, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMalloc(&d_seg_lengthes, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMallocHost(&h_keys_sizes, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMalloc(&d_keys_sizes, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMallocHost(&h_keys_sizes_new, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMalloc(&d_keys_sizes_new, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMallocHost(&update_add_size, num_blocks * sizeof(SIZE_TYPE)));
    cErr(cudaMallocHost(&h_need_sign_insert, num_blocks * sizeof(bool)));
    cErr(cudaMalloc(&d_need_sign_insert, num_blocks * sizeof(bool)));

    cErr(cudaMallocHost(&block_edge_num, sizeof(SIZE_TYPE) * num_blocks));
    for(int i = 0; i < num_blocks; i++) {
        SIZE_TYPE *lower_bound, *upper_bound;
        cErr(cudaMalloc(&lower_bound, sizeof(SIZE_TYPE) * 64)); // 这里假定层数最高不超过64层
        cErr(cudaMalloc(&upper_bound, sizeof(SIZE_TYPE) * 64));
        lower_boundes[i] = lower_bound;
        upper_boundes[i] = upper_bound;
        block_edge_num[i] = 2;
        h_need_sign_insert[i] = false;
    }
    cErr(cudaMemcpy(d_lower_boundes, lower_boundes, num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(d_upper_boundes, upper_boundes, num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(d_keyss, keyss, num_blocks * sizeof(KEY_TYPE*), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(d_valuess, valuess, num_blocks * sizeof(VALUE_TYPE*), cudaMemcpyHostToDevice));
    cErr(cudaDeviceSynchronize());
}

Multi_GPMA::~Multi_GPMA() {
    // A. Free per-block lower_boundes[i] and upper_boundes[i]
    if (lower_boundes) {
        for (SIZE_TYPE i = 0; i < num_blocks; i++) {
            if (lower_boundes[i]) cudaFree(lower_boundes[i]);
        }
    }
    if (upper_boundes) {
        for (SIZE_TYPE i = 0; i < num_blocks; i++) {
            if (upper_boundes[i]) cudaFree(upper_boundes[i]);
        }
    }

    // A. Free per-block keyss[i] and valuess[i] (allocated by malloc_multi_gpma)
    if (keyss) {
        for (SIZE_TYPE i = 0; i < num_blocks; i++) {
            if (keyss[i]) cudaFree(keyss[i]);
        }
    }
    if (valuess) {
        for (SIZE_TYPE i = 0; i < num_blocks; i++) {
            if (valuess[i]) cudaFree(valuess[i]);
        }
    }

    // B. Free per-block row_offset[i] and outDegree[i]
    // These are device-only pointer arrays — download to host to get each pointer.
    if (row_offset) {
        SIZE_TYPE **h_ptrs = nullptr;
        cudaMallocHost(&h_ptrs, num_blocks * sizeof(SIZE_TYPE*));
        if (h_ptrs) {
            cudaMemcpy(h_ptrs, row_offset, num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyDeviceToHost);
            for (SIZE_TYPE i = 0; i < num_blocks; i++) {
                if (h_ptrs[i]) cudaFree(h_ptrs[i]);
            }
            cudaFreeHost(h_ptrs);
        }
    }
    if (outDegree) {
        SIZE_TYPE **h_ptrs = nullptr;
        cudaMallocHost(&h_ptrs, num_blocks * sizeof(SIZE_TYPE*));
        if (h_ptrs) {
            cudaMemcpy(h_ptrs, outDegree, num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyDeviceToHost);
            for (SIZE_TYPE i = 0; i < num_blocks; i++) {
                if (h_ptrs[i]) cudaFree(h_ptrs[i]);
            }
            cudaFreeHost(h_ptrs);
        }
    }

    // C. Free top-level device arrays
    if (d_keyss)            cudaFree(d_keyss);
    if (d_valuess)          cudaFree(d_valuess);
    if (d_lower_boundes)    cudaFree(d_lower_boundes);
    if (d_upper_boundes)    cudaFree(d_upper_boundes);
    if (row_offset)         cudaFree(row_offset);
    if (outDegree)          cudaFree(outDegree);
    if (d_tree_heights)     cudaFree(d_tree_heights);
    if (d_seg_lengthes)     cudaFree(d_seg_lengthes);
    if (d_keys_sizes)       cudaFree(d_keys_sizes);
    if (d_keys_sizes_new)   cudaFree(d_keys_sizes_new);
    if (d_need_sign_insert) cudaFree(d_need_sign_insert);

    // D. Free top-level pinned-host arrays
    if (keyss)              cudaFreeHost(keyss);
    if (valuess)            cudaFreeHost(valuess);
    if (lower_boundes)      cudaFreeHost(lower_boundes);
    if (upper_boundes)      cudaFreeHost(upper_boundes);
    if (h_seg_lengthes)     cudaFreeHost(h_seg_lengthes);
    if (h_keys_sizes)       cudaFreeHost(h_keys_sizes);
    if (h_keys_sizes_new)   cudaFreeHost(h_keys_sizes_new);
    if (update_add_size)    cudaFreeHost(update_add_size);
    if (h_need_sign_insert) cudaFreeHost(h_need_sign_insert);
    if (block_edge_num)     cudaFreeHost(block_edge_num);
}

__host__
void resize_row_arrays(Multi_GPMA *gpma, SIZE_TYPE new_row_num) {
    if (new_row_num <= gpma->row_num) return;
    SIZE_TYPE old_row_num = gpma->row_num;
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM;

    // Download device pointer arrays to host
    SIZE_TYPE **h_row_offset;
    SIZE_TYPE **h_outDegree;
    cErr(cudaMallocHost(&h_row_offset, gpma->num_blocks * sizeof(SIZE_TYPE*)));
    cErr(cudaMallocHost(&h_outDegree, gpma->num_blocks * sizeof(SIZE_TYPE*)));
    cErr(cudaMemcpy(h_row_offset, gpma->row_offset, gpma->num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyDeviceToHost));
    cErr(cudaMemcpy(h_outDegree, gpma->outDegree, gpma->num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyDeviceToHost));

    for (SIZE_TYPE i = 0; i < gpma->num_blocks; i++) {
        // Resize row_offset[i]: old size = old_row_num + 1, new size = new_row_num + 1
        SIZE_TYPE *new_row_offset;
        cErr(cudaMalloc(&new_row_offset, sizeof(SIZE_TYPE) * (new_row_num + 1)));
        // Copy old data
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, old_row_num + 1);
        memcpy_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(new_row_offset, h_row_offset[i], old_row_num + 1);
        // Zero new portion
        SIZE_TYPE zero_count = new_row_num - old_row_num;
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, zero_count);
        memset_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(new_row_offset + old_row_num + 1, 0, zero_count);
        cErr(cudaFree(h_row_offset[i]));
        h_row_offset[i] = new_row_offset;

        // Resize outDegree[i]: old size = old_row_num, new size = new_row_num
        SIZE_TYPE *new_out_degree;
        cErr(cudaMalloc(&new_out_degree, sizeof(SIZE_TYPE) * new_row_num));
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, old_row_num);
        memcpy_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(new_out_degree, h_outDegree[i], old_row_num);
        zero_count = new_row_num - old_row_num;
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, zero_count);
        memset_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(new_out_degree + old_row_num, 0, zero_count);
        cErr(cudaFree(h_outDegree[i]));
        h_outDegree[i] = new_out_degree;
    }

    // Upload updated pointer arrays back to device
    cErr(cudaMemcpy(gpma->row_offset, h_row_offset, gpma->num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyHostToDevice));
    cErr(cudaMemcpy(gpma->outDegree, h_outDegree, gpma->num_blocks * sizeof(SIZE_TYPE*), cudaMemcpyHostToDevice));
    cErr(cudaDeviceSynchronize());
    cudaFreeHost(h_row_offset);
    cudaFreeHost(h_outDegree);

    gpma->row_num = new_row_num;
}
