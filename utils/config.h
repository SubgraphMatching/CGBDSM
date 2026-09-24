#pragma once
#include <cstdint>

#define NOT_EXIST UINT32_MAX
#define MAX_VCOUNT 12u
#define MAX_ECOUNT 42u
#define MIN_NBR_SIZE 8u

#define GRID_DIM 1024u
#define BLOCK_DIM 512u
#define WARP_SIZE 32u
#define NWARP_PER_BLOCK (BLOCK_DIM / WARP_SIZE)

#define UNROLL 8u
// #define USE_GLOBAL_RQ
// #define USE_PUSH_VERIFY
// #define USE_PUSH_ALL

#define NBR_SPACE (256ul * 1024 * 1024 / sizeof(uint32_t)) // 256 MB of uint32_t
#define RES_SPACE (4ul * 1024 * 1024 * 1024 / sizeof(uint32_t)) // 4 GB of uint32_t
#define SIZE_SPACE (2ul * 1024 * 1024 * 1024 / sizeof(long)) // 2 GB of long
#define MIN_NRESULTS_TO_GPU (1 << 16)
// #define FORCE_UPDATE_EDGE_START
// #define ENABLE_WARP_STEALING

// Matching order strategy: GSI score-based (requires USE_MERGED_MATCHING)
// #define USE_GSI_ORDER

// Work stealing tuning parameters (only effective when ENABLE_WARP_STEALING is defined)
#define EXPOSURE_THRESHOLD 32u   // Min neighbor count to publish exposure (RDMCE recommends tau=24)
#define STEAL_CHUNK        32u   // Neighbors per thief claim (= WARP_SIZE for full warp utilization)

// Support Mask Configuration: bit-width per vertex for false-positive reduction
// Set to 1, 4, 8, 16, 32, or 64.
//   1: Packed boolean bitmap (32 vertices per uint32_t), no bloom filter
//   4/8/16: Reserved for future (kernel code not yet implemented)
//   32/64: Bloom filter per vertex
#define SUPPORT_MASK_WIDTH 1

#if SUPPORT_MASK_WIDTH == 1
    // Packed boolean: 32 vertices packed into one uint32_t, each bit = one vertex active
    typedef uint32_t smask_t;
    #define SMASK_ZERO 0u
    #define SMASK_ALL 0xFFFFFFFFu
    #define SMASK_DV_STRIDE(n) (((n) + 31u) / 32u)  // words per (edge, qv)
    #define SMASK_IDX(dv) ((dv) >> 5)                 // word index for vertex dv
    #define SMASK_BIT(dv) (1u << ((dv) & 31))         // bit mask for vertex dv
    #define SMASK_TEST(ptr, dv) ((ptr)[(dv) >> 5] & (1u << ((dv) & 31)))
    #define USE_CUM_PATH_MASK 0
#elif SUPPORT_MASK_WIDTH == 4
    // Reserved: 8 vertices per uint32_t (4 bits each)
    typedef uint32_t smask_t;
    #define SMASK_ZERO 0u
    #define SMASK_ALL 0xFFFFFFFFu
    #define SMASK_DV_STRIDE(n) (((n) + 7u) / 8u)
    #define SMASK_IDX(dv) ((dv) >> 3)
    #define SMASK_BIT(dv) (0xFu << (4u * ((dv) & 7u)))
    #define SMASK_TEST(ptr, dv) ((ptr)[(dv) >> 3] & (0xFu << (4u * ((dv) & 7u))))
    #define USE_CUM_PATH_MASK 0
    #error "SUPPORT_MASK_WIDTH=4 kernel code not yet implemented"
#elif SUPPORT_MASK_WIDTH == 8
    typedef uint8_t smask_t;
    #define SMASK_ZERO 0u
    #define SMASK_ALL 0xFFu
    #define SMASK_DV_STRIDE(n) (n)
    #define SMASK_IDX(dv) (dv)
    #define SMASK_BIT(dv) (0u)
    #define SMASK_TEST(ptr, dv) ((ptr)[dv] != SMASK_ZERO)
    #define USE_CUM_PATH_MASK 1
#elif SUPPORT_MASK_WIDTH == 16
    typedef unsigned short smask_t;
    #define SMASK_ZERO 0u
    #define SMASK_ALL 0xFFFFu
    #define SMASK_DV_STRIDE(n) (n)
    #define SMASK_IDX(dv) (dv)
    #define SMASK_BIT(dv) (0u)
    #define SMASK_TEST(ptr, dv) ((ptr)[dv] != SMASK_ZERO)
    #define USE_CUM_PATH_MASK 1
#elif SUPPORT_MASK_WIDTH == 32
    typedef uint32_t smask_t;
    #define SMASK_ZERO 0u
    #define SMASK_ALL 0xFFFFFFFFu
    #define SMASK_DV_STRIDE(n) (n)
    #define SMASK_IDX(dv) (dv)
    #define SMASK_BIT(dv) (0u)
    #define SMASK_TEST(ptr, dv) ((ptr)[dv] != SMASK_ZERO)
    #define USE_CUM_PATH_MASK 1
#elif SUPPORT_MASK_WIDTH == 64
    typedef unsigned long long smask_t;
    #define SMASK_ZERO 0ull
    #define SMASK_ALL 0xFFFFFFFFFFFFFFFFull
    #define SMASK_DV_STRIDE(n) (n)
    #define SMASK_IDX(dv) (dv)
    #define SMASK_BIT(dv) (0u)
    #define SMASK_TEST(ptr, dv) ((ptr)[dv] != SMASK_ZERO)
    #define USE_CUM_PATH_MASK 1
#else
    #error "SUPPORT_MASK_WIDTH must be 1, 4, 8, 16, 32, or 64"
#endif

// Enumeration helper functions (device inline, for both packed and non-packed access)
// Only compiled in CUDA code (not host C++)
#ifdef __CUDACC__
#if SUPPORT_MASK_WIDTH == 1
static __forceinline__ __device__ uint32_t smask_read(const smask_t* __restrict__ ptr, uint32_t dv) {
    return ptr[dv >> 5] & (1u << (dv & 31));
}
static __forceinline__ __device__ bool smask_pair_ok(uint32_t v_active, const smask_t* __restrict__ nbr_ptr, uint32_t nbr) {
    return v_active && (nbr_ptr[nbr >> 5] & (1u << (nbr & 31)));
}
#else
static __forceinline__ __device__ smask_t smask_read(const smask_t* __restrict__ ptr, uint32_t dv) {
    return ptr[dv];
}
static __forceinline__ __device__ bool smask_pair_ok(smask_t v_sm, const smask_t* __restrict__ nbr_ptr, uint32_t nbr) {
    smask_t nbr_sm = nbr_ptr[nbr];
    return nbr_sm != SMASK_ZERO && (v_sm & nbr_sm) != SMASK_ZERO;
}
#endif
#endif // __CUDACC__
