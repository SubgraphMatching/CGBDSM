#ifndef UTILS_CUDA_HELPERS_H
#define UTILS_CUDA_HELPERS_H

#include <stdio.h>
#include <cstdint>
#include <chrono>


#define DIV_CEIL(a,b) ((a) / (b) + ((a) % (b) != 0))

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define cucheck_dev(call)                                           \
{                                                                   \
    cudaError_t cucheck_err = (call);                               \
    if(cucheck_err != cudaSuccess) {                                \
        const char *err_str = cudaGetErrorString(cucheck_err);      \
        printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);       \
        assert(0);                                                  \
    }                                                               \
}

#define CUB(code)                                                   \
{                                                                   \
    void *pre_d_temp_storage_ = d_temp_storage_;                    \
    d_temp_storage_ = NULL;                                         \
    code;                                                           \
    if (temp_storage_bytes_ > temp_storage_capability_)             \
    {                                                               \
        if (temp_storage_capability_ != 0ul)                        \
        {                                                           \
            cudaErrorCheck(cudaFree(pre_d_temp_storage_));          \
        }                                                           \
        temp_storage_bytes_ = temp_storage_capability_ =            \
            (size_t)exp2(ceil(log2(temp_storage_bytes_)));          \
        cudaErrorCheck(cudaMalloc(                                  \
            &d_temp_storage_, temp_storage_bytes_));                \
    }                                                               \
    else                                                            \
    {                                                               \
        d_temp_storage_ = pre_d_temp_storage_;                      \
    }                                                               \
    code;                                                           \
}

#define CUB1(code)                                                  \
{                                                                   \
    d_temp_storage_ = NULL;                                         \
    code;                                                           \
    cudaErrorCheck(cudaMalloc(                                      \
        &d_temp_storage_, temp_storage_bytes_));                    \
    code;                                                           \
    cudaErrorCheck(cudaFree(d_temp_storage_));                      \
}

#define ReAlloc(var, min_size, capability, type)                    \
{                                                                   \
    if (min_size > capability)                                      \
    {                                                               \
        if (capability != 0) cudaErrorCheck(cudaFree(var));         \
        size_t new_size = max(                                      \
            (size_t)exp2(ceil(log2(min_size))), 8ul);               \
        cudaErrorCheck(cudaMalloc(&var, sizeof(type) * new_size));  \
        capability = new_size;                                      \
    }                                                               \
}


#define TIME_INIT() cudaEvent_t cuda_start, cuda_end;               \
    float kernel_time;                                              \
    auto start_clock = std::chrono::high_resolution_clock::now();   \
    auto end_clock = std::chrono::high_resolution_clock::now();     \
    std::chrono::duration<double> diff = end_clock - start_clock

#define TIME_START()                                                \
    start_clock = std::chrono::high_resolution_clock::now();        \
    cudaEventCreate(&cuda_start);                                   \
    cudaEventCreate(&cuda_end);                                     \
    cudaEventRecord(cuda_start)

#define TIME_END()                                                  \
    end_clock = std::chrono::high_resolution_clock::now();          \
    cudaEventRecord(cuda_end);                                      \
    cudaEventSynchronize(cuda_start);                               \
    cudaEventSynchronize(cuda_end);                                 \
    cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);       \
    diff = end_clock - start_clock;

#define PRINT_LOCAL_TIME(name)                                      \
    std::cout << name << ", time (ms): " <<                         \
        static_cast<unsigned long>(diff.count() * 1000) <<          \
        "(host), " << static_cast<unsigned long>(kernel_time) <<    \
        "(kernel)\n"


#define LTIME_INIT()                                                \
    auto lstart_clock = std::chrono::high_resolution_clock::now();  \
    auto lend_clock = std::chrono::high_resolution_clock::now();    \
    std::chrono::duration<double> ldiff = lend_clock - lstart_clock

#define LTIME_START()                                               \
    lstart_clock = std::chrono::high_resolution_clock::now();

#define LTIME_END()                                                 \
    lend_clock = std::chrono::high_resolution_clock::now();         \
    ldiff = lend_clock - lstart_clock;

#define LPRINT_LOCAL_TIME(name)                                     \
    std::cout << name << ", time (ms): " <<                         \
        static_cast<unsigned long>(ldiff.count() * 1000) <<         \
        "(host)\n"

#define RTIME_INIT()                                                \
    auto rstart_clock = std::chrono::high_resolution_clock::now();  \
    auto rend_clock = std::chrono::high_resolution_clock::now();    \
    std::chrono::duration<double> rdiff = rend_clock - rstart_clock

#define RTIME_START()                                               \
    rstart_clock = std::chrono::high_resolution_clock::now();

#define RTIME_END()                                                 \
    rend_clock = std::chrono::high_resolution_clock::now();         \
    rdiff = rend_clock - rstart_clock;

#define RPRINT_LOCAL_TIME(name)                                     \
    std::cout << name << ", time (ms): " <<                         \
        static_cast<unsigned long>(rdiff.count() * 1000) <<         \
        "(host)\n"

#endif //UTILS_CUDA_HELPERS_H
