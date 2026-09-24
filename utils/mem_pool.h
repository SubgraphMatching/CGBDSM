#ifndef UTILS_MEM_POOL
#define UTILS_MEM_POOL

#include <cstdint>
#include "utils/cuda_helpers.h"

template<typename T>
struct MemPool
{
    T *array_;
    unsigned long long int capability_;
    unsigned long long int h_occupy_;
    unsigned long long int *occupy_;

    void Alloc(unsigned long long int size)
    {
        cudaErrorCheck(cudaMalloc(&array_, sizeof(T) * size));
        capability_ = size;
        h_occupy_ = 0ul;
        cudaErrorCheck(cudaMalloc(&occupy_, sizeof(unsigned long long int)));
        cudaErrorCheck(cudaMemset(occupy_, 0u, sizeof(unsigned long long int)));
    }
    void Free()
    {
        cudaErrorCheck(cudaFree(array_));
    }
    void Reset()
    {
        cudaErrorCheck(cudaMemset(occupy_, 0u, sizeof(unsigned long long int)));
    }
    bool OutOfMemory()
    {
        cudaErrorCheck(cudaMemcpy(&h_occupy_, occupy_, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        return h_occupy_ >= capability_;
    }
};

template<typename T>
struct CyclicQueue
{
    T *array_;
    unsigned long long int capability_;
    unsigned long long int available_start_;
    unsigned long long int available_end_;

    void Alloc(size_t capability)
    {
        cudaErrorCheck(cudaMalloc(&array_, sizeof(T) * capability));
        capability_ = capability;
        available_start_ = 0;
        available_end_ = capability;
    }
    void Free()
    {
        cudaErrorCheck(cudaFree(array_));
    }
    void Reset()
    {
        available_start_ = 0;
        available_end_ = capability_;
    }
    unsigned long long int TryMax()
    {
        return available_start_;
    }
    size_t GetFree()
    {
        return (available_end_ + capability_ - available_start_) % capability_;
    }
    void Push(size_t size)
    {
        available_start_ = (available_start_ + size) % capability_;
    }
    void Pop(size_t size)
    {
        available_end_ = (available_end_ + size) % capability_;
    }
};

#endif