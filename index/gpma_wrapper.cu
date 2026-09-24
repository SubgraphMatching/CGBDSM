#include "index/gpma_wrapper.cuh"
#include "utils/cuda_helpers.h"
#include <cub/cub.cuh>
#include <cstdio>

void CGCSM_GPMA::ensureUpdateBuffer(uint32_t needed)
{
    ReAlloc(gpma_->update_keys, needed, cap_update_keys_, KEY_TYPE);
    ReAlloc(gpma_->update_values, needed, cap_update_values_, VALUE_TYPE);
    ReAlloc(gpma_->update_nodes, needed, cap_update_nodes_, KEY_TYPE);
    ReAlloc(gpma_->unique_update_nodes, needed, cap_unique_update_nodes_, KEY_TYPE);
    ReAlloc(gpma_->update_offset, needed + 1, cap_update_offset_, SIZE_TYPE);
    ReAlloc(gpma_->tmp_keys_array, needed, cap_tmp_keys_, KEY_TYPE);
    ReAlloc(gpma_->tmp_values_array, needed, cap_tmp_values_, VALUE_TYPE);
    ReAlloc(gpma_->tmp_label_array, needed, cap_tmp_label_, KEY_TYPE);
    ReAlloc(gpma_->tmp_exscan_array, needed, cap_tmp_exscan_, KEY_TYPE);
}

void CGCSM_GPMA::Init(uint32_t num_edges, uint32_t max_vertices)
{
    num_edges_ = num_edges;
    max_vertices_ = max_vertices;
    gpma_ = new Multi_GPMA(max_vertices, num_edges);
}

void CGCSM_GPMA::BulkLoad(const uint64_t *d_all_keys, uint32_t total_count,
                          const uint32_t *counts_per_edge)
{
    if (total_count == 0)
        return;

    uint32_t guard_count = max_vertices_ * num_edges_;
    uint32_t total_with_guards = total_count + guard_count;
    std::cout << "total_count" << total_count << " " << "max_vertices_" << max_vertices_ << "num_edges_" << num_edges_ << std::endl;

    ensureUpdateBuffer(total_with_guards);
    cudaErrorCheck(cudaMemcpy(gpma_->update_keys, d_all_keys, total_count * sizeof(KEY_TYPE), cudaMemcpyDeviceToDevice));
    cudaErrorCheck(cudaMemset(gpma_->update_values, 1, total_count * sizeof(VALUE_TYPE)));

    for (uint32_t e = 0; e < num_edges_; e++)
        gpma_->block_edge_num[e] = counts_per_edge[e] + max_vertices_ + 2;

    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, max_vertices_);
    init_row_wall<<<BLOCKS_NUM, THREADS_NUM>>>(
        gpma_->update_keys + total_count, max_vertices_, num_edges_);
    memset_kernel<VALUE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(
        gpma_->update_values + total_count, (VALUE_TYPE)1, guard_count);
    gpma_->update_keys_size = total_with_guards;

    resize_gpmas_batch(gpma_);
    init_gpmas_keys_values_batch(gpma_);
    update_gpma_stage1(gpma_);
    update_gpma_stage2(gpma_);
    update_gpma_stage3(gpma_);
    update_gpma_stage4(gpma_);
    cudaErrorCheck(cudaDeviceSynchronize());
}

void CGCSM_GPMA::BatchUpdate(uint64_t *d_ins_keys, uint32_t ins_count,
                             uint64_t *d_del_keys, uint32_t del_count,
                             int32_t *block_delta,
                             uint32_t new_max_vertices)
{
    uint32_t total = ins_count + del_count;

    uint32_t guard_count = 0;
    if (new_max_vertices > max_vertices_)
    {
        uint32_t old_max = max_vertices_;
        resize_row_arrays(gpma_, new_max_vertices);

        guard_count = (new_max_vertices - old_max) * num_edges_;
        uint32_t total_with_guards = guard_count + total;
        ensureUpdateBuffer(total_with_guards);

        SIZE_TYPE THREADS_NUM = 32;
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, new_max_vertices - old_max);
        init_row_wall_offset<<<BLOCKS_NUM, THREADS_NUM>>>(
            gpma_->update_keys + total, old_max, new_max_vertices - old_max, num_edges_);
        memset_kernel<VALUE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(
            gpma_->update_values + total, (VALUE_TYPE)1, guard_count);

        for (uint32_t e = 0; e < num_edges_; e++)
            gpma_->block_edge_num[e] += (new_max_vertices - old_max);

        max_vertices_ = new_max_vertices;
    }
    else
    {
        if (total == 0)
            return;
        ensureUpdateBuffer(total);
    }

    if (ins_count > 0)
        cudaErrorCheck(cudaMemcpy(gpma_->update_keys, d_ins_keys, ins_count * sizeof(KEY_TYPE), cudaMemcpyDeviceToDevice));
    if (del_count > 0)
        cudaErrorCheck(cudaMemcpy(gpma_->update_keys + ins_count, d_del_keys, del_count * sizeof(KEY_TYPE), cudaMemcpyDeviceToDevice));
    cudaErrorCheck(cudaMemset(gpma_->update_values, 1, ins_count * sizeof(VALUE_TYPE)));
    if (del_count > 0)
        cudaErrorCheck(cudaMemset(gpma_->update_values + ins_count, 0, del_count * sizeof(VALUE_TYPE)));

    if (block_delta)
        for (uint32_t e = 0; e < num_edges_; e++)
            gpma_->block_edge_num[e] += block_delta[e];
    gpma_->update_keys_size = total + guard_count;

    update_gpma_stage1(gpma_);
    resize_gpmas_batch(gpma_);
    update_gpma_stage2(gpma_);
    gpma_->level = 0;
    update_gpma_stage3(gpma_);
    update_gpma_stage4(gpma_);
    cudaErrorCheck(cudaDeviceSynchronize());
}

void CGCSM_GPMA::Destroy()
{
    if (gpma_)
    {
        // Free update buffers allocated by ensureUpdateBuffer (via ReAlloc)
        if (cap_update_keys_ > 0)         cudaFree(gpma_->update_keys);
        if (cap_update_values_ > 0)       cudaFree(gpma_->update_values);
        if (cap_update_nodes_ > 0)        cudaFree(gpma_->update_nodes);
        if (cap_unique_update_nodes_ > 0) cudaFree(gpma_->unique_update_nodes);
        if (cap_update_offset_ > 0)       cudaFree(gpma_->update_offset);
        if (cap_tmp_keys_ > 0)            cudaFree(gpma_->tmp_keys_array);
        if (cap_tmp_values_ > 0)          cudaFree(gpma_->tmp_values_array);
        if (cap_tmp_label_ > 0)           cudaFree(gpma_->tmp_label_array);
        if (cap_tmp_exscan_ > 0)          cudaFree(gpma_->tmp_exscan_array);

        // Reset capacities so double-Destroy is safe
        cap_update_keys_ = 0;
        cap_update_values_ = 0;
        cap_update_nodes_ = 0;
        cap_unique_update_nodes_ = 0;
        cap_update_offset_ = 0;
        cap_tmp_keys_ = 0;
        cap_tmp_values_ = 0;
        cap_tmp_label_ = 0;
        cap_tmp_exscan_ = 0;

        delete gpma_;
        gpma_ = nullptr;
    }
}
