#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>

typedef unsigned long long KEY_TYPE; // from to 
typedef bool VALUE_TYPE;
typedef unsigned int SIZE_TYPE;

typedef thrust::device_vector<KEY_TYPE> DEV_VEC_KEY;
typedef thrust::device_vector<VALUE_TYPE> DEV_VEC_VALUE;
typedef thrust::device_vector<SIZE_TYPE> DEV_VEC_SIZE;

using HOST_VEC_KEY = thrust::host_vector<KEY_TYPE, thrust::mr::stateless_resource_allocator<KEY_TYPE, thrust::universal_host_pinned_memory_resource>>;
using HOST_VEC_VALUE = thrust::host_vector<VALUE_TYPE, thrust::mr::stateless_resource_allocator<VALUE_TYPE, thrust::universal_host_pinned_memory_resource>>;
using HOST_VEC_SIZE = thrust::host_vector<SIZE_TYPE, thrust::mr::stateless_resource_allocator<SIZE_TYPE, thrust::universal_host_pinned_memory_resource>>;

typedef KEY_TYPE* KEY_PTR;
typedef VALUE_TYPE* VALUE_PTR;

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())

const KEY_TYPE KEY_NONE = 0xFFFFFFFFFFFFFFFF;
const KEY_TYPE KEY_MAX = 0xFFFFFFFFFFFFFFFE;
const SIZE_TYPE SIZE_NONE = 0xFFFFFFFF;
const VALUE_TYPE VALUE_NONE = 0;
const KEY_TYPE COL_IDX_NONE = 0xFFFFFFFF;

const SIZE_TYPE MAX_BLOCKS_NUM = 96 * 8;
#define CALC_BLOCKS_NUM(ITEMS_PER_BLOCK, CALC_SIZE) min(MAX_BLOCKS_NUM, (CALC_SIZE - 1) / ITEMS_PER_BLOCK + 1)

class Multi_GPMA {
public:
    SIZE_TYPE num_blocks;
    SIZE_TYPE row_num; // 顶点数目
    KEY_TYPE** keyss = nullptr;
    VALUE_TYPE** valuess = nullptr;
    KEY_TYPE** d_keyss = nullptr;
    VALUE_TYPE** d_valuess = nullptr;
    SIZE_TYPE** row_offset = nullptr; // [num_blocks][row_num + 1]
    SIZE_TYPE** outDegree = nullptr; // 统计出度, [num_blocks][row_num]
    SIZE_TYPE* d_tree_heights = nullptr;
    SIZE_TYPE* h_seg_lengthes = nullptr;
    SIZE_TYPE* d_seg_lengthes = nullptr;
    SIZE_TYPE* h_keys_sizes = nullptr;
    SIZE_TYPE* d_keys_sizes = nullptr;
    SIZE_TYPE* h_keys_sizes_new = nullptr;
    SIZE_TYPE* d_keys_sizes_new = nullptr;
    double mem_usage = 0;

    double density_lower_thres_leaf = 0.08;
    double density_lower_thres_root = 0.15; // 这里防止出现不连续的情况
    double density_upper_thres_root = 0.84;
    double density_upper_thres_leaf = 0.92;
    SIZE_TYPE** lower_boundes = nullptr;
    SIZE_TYPE** upper_boundes = nullptr;
    SIZE_TYPE** d_lower_boundes = nullptr;
    SIZE_TYPE** d_upper_boundes = nullptr;

    SIZE_TYPE *block_edge_num = nullptr;
    SIZE_TYPE* update_add_size = nullptr; // 统计加边的数量方便进行判断是否sign

    KEY_TYPE *update_keys = nullptr;
    VALUE_TYPE *update_values = nullptr;
    KEY_TYPE *update_nodes = nullptr;
    KEY_TYPE *unique_update_nodes = nullptr;
    SIZE_TYPE *update_offset = nullptr; // 注意cudamalloc keys.length + 1
    SIZE_TYPE update_keys_size = 0;
    bool* d_need_sign_insert = nullptr;
    bool* h_need_sign_insert = nullptr;

    KEY_TYPE *tmp_keys_array = nullptr;
    VALUE_TYPE *tmp_values_array = nullptr;
    KEY_TYPE *tmp_label_array = nullptr;
    KEY_TYPE *tmp_exscan_array = nullptr;

    SIZE_TYPE update_size;
    SIZE_TYPE unique_node_size;
    SIZE_TYPE compacted_size;
    SIZE_TYPE update_size_update_width;
    SIZE_TYPE update_width;
    SIZE_TYPE level;

    SIZE_TYPE node_per_block;
    SIZE_TYPE degree_upper;

    KEY_TYPE all_block_update_size = 0;
    KEY_TYPE all_block_valid_update_size = 0;

    KEY_TYPE all_kernel_update_size = 0;
    KEY_TYPE all_kernel_valid_update_size = 0;

    SIZE_TYPE all_kernel_update_time = 0;
    SIZE_TYPE all_block_update_time = 0;

    KEY_TYPE single_kernel_update_size = 0;
    KEY_TYPE single_block_update_size = 0;

    float rebalance_block_batch_time = 0;
    float rebalance_kernel_batch_time = 0;

    float single_rebalance_block_batch_time = 0;
    float single_rebalance_kernel_batch_time = 0;

    Multi_GPMA(SIZE_TYPE row_num_, SIZE_TYPE num_blocks_);
    ~Multi_GPMA();
};

__global__
void init_row_wall(KEY_TYPE *data, SIZE_TYPE size, SIZE_TYPE num_blocks);

__global__
void init_row_wall_offset(KEY_TYPE *data, SIZE_TYPE start, SIZE_TYPE count, SIZE_TYPE num_blocks);

__host__
void resize_row_arrays(Multi_GPMA *gpma, SIZE_TYPE new_row_num);

template<typename T>
__global__
void memset_kernel(T *data, T value, SIZE_TYPE size);

__host__
void update_gpma_stage1(Multi_GPMA *gpma);

__host__
void resize_gpmas_batch(Multi_GPMA *gpma);

__host__
void init_gpmas_keys_values_batch(Multi_GPMA *gpma);

__host__
void update_gpma_stage2(Multi_GPMA *gpma);

__host__
void update_gpma_stage3(Multi_GPMA *gpma);

__host__
void update_gpma_stage4(Multi_GPMA *gpma);
