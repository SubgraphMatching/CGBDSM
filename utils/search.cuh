#ifndef UTILS_SEARCH_CUH
#define UTILS_SEARCH_CUH

#include <cstdint>
#include "config.h"

template<typename T1, typename T2>
__forceinline__ __device__ T2 lower_bound(const T1* array, const T2 size, const T1 v)
{
    if (array == NULL || size == 0 || array[size - 1] < v) return size;

    T2 low = 0u, high = size - 1, mid = (low + high) / 2;
    while (low < high)
    {
        if (array[mid] < v)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

template<typename T>
__forceinline__ __device__ void load_lb_cache(T* cache, const T* array, const uint32_t size)
{
    int lane = threadIdx.x & (WARP_SIZE - 1);
    if (size > 0)
    {
        cache[lane] = array[static_cast<uint32_t>(static_cast<uint64_t>(lane) * size / WARP_SIZE)];
    }
    __syncwarp();
}

template<typename T>
__forceinline__ __device__ T lower_bound_2phase(const T* array, const uint32_t size, const T v, const T* cache)
{
    if (size == 0 || array[size - 1] < v) return size;

    int bottom = 0, top = WARP_SIZE;
    while (top > bottom + 1)
    {
        int mid = (top + bottom) / 2;
        if (cache[mid] < v) bottom = mid;
        else top = mid;
    }

    T low  = static_cast<T>(static_cast<uint64_t>(bottom) * size / WARP_SIZE);
    T high = min(static_cast<T>(static_cast<uint64_t>(top) * size / WARP_SIZE), size);
    while (low < high)
    {
        T mid = (low + high) / 2;
        if (array[mid] < v) low = mid + 1;
        else high = mid;
    }
    return low;
}

template<typename T1, typename T2>
__forceinline__ __device__ T2 upper_bound(const T1* array, const T2 size, const T1 v)
{
    T2 count = size, step, first = 0, mid;

    while (count > 0)
    {
        step = count / 2;
        mid = first + step;
        if (array[mid] <= v)
        {
            first = mid + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }
    return first;
}

#endif