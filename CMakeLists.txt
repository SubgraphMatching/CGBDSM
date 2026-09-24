cmake_minimum_required(VERSION 3.10)
project(cgcsm LANGUAGES CUDA CXX)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/local/cuda-12.9/include)

add_subdirectory(utils/nucleus nd)

add_library(link STATIC
    graph/graph_gpu.h graph/graph_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/enumeration.h kernels/enumeration.cu
    kernels/enumeration_balance.h kernels/enumeration_balance.cu
    kernels/enumeration_dfs.cu kernels/enumeration_bfs.cu
    kernels/cartesian_product.h kernels/cartesian_product.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h)

set_target_properties(link PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON)

add_library(gpma_lib STATIC
    gpma/gpma.cu
    index/gpma_wrapper.cuh index/gpma_wrapper.cu)

set_target_properties(gpma_lib PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(gpma_lib PRIVATE CUDA_FORCE_CDP1_IF_SUPPORTED)

add_executable(cgcsm
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    #index/cpu_pre_matcher.h index/cpu_pre_matcher.cpp
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_options(link PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:Debug>:-G -g>
    $<$<CONFIG:Release>:-lineinfo -O3>
        --ptxas-options=-v
    >
)

target_compile_options(cgcsm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

set(TBB_ROOT "/opt/intel/oneapi/tbb/2022.3")
find_package(TBB REQUIRED PATHS ${TBB_ROOT} NO_DEFAULT_PATH)

target_link_libraries(cgcsm PUBLIC
    link
    # gpma_lib
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

# 带 SKIP_BUILD_LOCAL_INDEX 宏的 link 变体
add_library(link_nolocal STATIC
    graph/graph_gpu.h graph/graph_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/enumeration.h kernels/enumeration.cu
    kernels/enumeration_balance.h kernels/enumeration_balance.cu
    kernels/enumeration_dfs.cu kernels/enumeration_bfs.cu
    kernels/cartesian_product.h kernels/cartesian_product.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h)

set_target_properties(link_nolocal PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(link_nolocal PRIVATE
    SKIP_BUILD_LOCAL_INDEX
)

target_compile_options(link_nolocal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:-lineinfo -O3>
        --ptxas-options=-v
    >
)


# 带 SKIP_BUILD_LOCAL_INDEX 宏的 link 变体
add_library(link_global STATIC
    graph/graph_gpu.h graph/graph_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/enumeration.h kernels/enumeration.cu
    kernels/enumeration_balance.h kernels/enumeration_balance.cu
    kernels/enumeration_dfs.cu kernels/enumeration_bfs.cu
    kernels/cartesian_product.h kernels/cartesian_product.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h)

set_target_properties(link_global PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(link_global PRIVATE
    USE_VALID_BITS
    USE_MERGED_MATCHING
    USE_GLOBAL_RQ
)

target_compile_options(link_global PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:-lineinfo -O3>
        --ptxas-options=-v
    >
)

# 跳过 BuildLocalIndex 的版本，用于性能对比
add_executable(cgcsm_without_localindex
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_without_localindex PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_without_localindex PRIVATE
    SKIP_BUILD_LOCAL_INDEX
)

target_compile_options(cgcsm_without_localindex PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_without_localindex PUBLIC
    link_nolocal
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

# Valid bit variant: uses sparse per-vertex valid bits instead of dense local_index
add_executable(cgcsm_valid_bits
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_valid_bits PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_valid_bits PRIVATE
    USE_VALID_BITS
)

target_compile_options(cgcsm_valid_bits PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_valid_bits PUBLIC
    link
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

# Merged matching variant: all edges share one kernel launch with 5-bit edge tag
add_library(link_merged STATIC
    graph/graph_gpu.h graph/graph_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/enumeration.h kernels/enumeration.cu
    kernels/enumeration_balance.h kernels/enumeration_balance.cu
    kernels/enumeration_dfs.cu kernels/enumeration_bfs.cu
    kernels/cartesian_product.h kernels/cartesian_product.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h)

set_target_properties(link_merged PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(link_merged PRIVATE
    USE_VALID_BITS
    USE_MERGED_MATCHING
)
# RELAX_DEBUG must be visible to link_merged too: graph_gpu.cu/enumeration.cu
# (RelaxDbgPrintSkipFail, g_dbg_skip_fail, the relax site) live here, not in the exe.
# option(CGCSM_RELAX_DEBUG "Enable relax skip-fail debug instrumentation (prints samples where relax skips a binary search that would fail)" OFF)
# if(CGCSM_RELAX_DEBUG)
#     target_compile_definitions(link_merged PRIVATE RELAX_DEBUG)
# endif()

target_compile_options(link_merged PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:-lineinfo -O3>
        --ptxas-options=-v
    >
)

# link_cpu_dfs: same as link_merged but with ENABLE_CPU_DFS so graph_gpu.cu's
# InitCPUDfs / Step5CPUGPUSplit are compiled (they are guarded by that flag).
add_library(link_cpu_dfs STATIC
    graph/graph_gpu.h graph/graph_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/enumeration.h kernels/enumeration.cu
    kernels/enumeration_balance.h kernels/enumeration_balance.cu
    kernels/enumeration_dfs.cu kernels/enumeration_bfs.cu
    kernels/cartesian_product.h kernels/cartesian_product.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h)

set_target_properties(link_cpu_dfs PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(link_cpu_dfs PRIVATE
    USE_VALID_BITS
    USE_MERGED_MATCHING
    ENABLE_CPU_DFS
)

target_compile_options(link_cpu_dfs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:-lineinfo -O3>
        --ptxas-options=-v
    >
)

add_executable(cgcsm_merged_matching
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_merged_matching PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_merged_matching PRIVATE
    USE_VALID_BITS
    USE_MERGED_MATCHING
)

# if(CGCSM_RELAX_DEBUG)
#     target_compile_definitions(cgcsm_merged_matching PRIVATE RELAX_DEBUG)
# endif()

target_compile_options(cgcsm_merged_matching PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_merged_matching PUBLIC
    link_merged
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

# GPMA variant: unified Multi_GPMA with merged matching
add_library(link_gpma STATIC
    graph/graph_gpu.h graph/graph_gpu.cu
    kernels/indexing.h kernels/indexing.cu
    kernels/enumeration.h kernels/enumeration.cu
    kernels/enumeration_balance.h kernels/enumeration_balance.cu
    kernels/enumeration_dfs.cu kernels/enumeration_bfs.cu
    kernels/cartesian_product.h kernels/cartesian_product.cu
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/cuda_helpers.h)

set_target_properties(link_gpma PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(link_gpma PRIVATE
    USE_GPMA
    USE_VALID_BITS
    USE_MERGED_MATCHING
)

target_compile_options(link_gpma PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:-lineinfo -O3>
        --ptxas-options=-v
    >
)

add_executable(cgcsm_gpma
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gpma.h index/calig_helper_gpma.cu
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_gpma PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_gpma PRIVATE
    USE_GPMA
    USE_VALID_BITS
    USE_MERGED_MATCHING
)

target_compile_options(cgcsm_gpma PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_gpma PUBLIC
    link_gpma
    gpma_lib
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

# Standalone graph sampler (no CUDA/TBB dependency)

# GAMMA-style per-vertex sorted merge variant
add_executable(cgcsm_gamma
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_gamma PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_gamma PRIVATE
    USE_GAMMA
    USE_VALID_BITS
    USE_MERGED_MATCHING
)

target_compile_options(cgcsm_gamma PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_gamma PUBLIC
    link_merged
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

# cgcsm_cpu_dfs: cgcsm_gamma + a host-side (NUMA-2) mirror of the index + a CPU
# port of extendBFSAllBit, so the CPU can share the final-DFS workload. Phase A
# ships a single-threaded full-frontier count cross-checked against the GPU.
add_executable(cgcsm_cpu_dfs
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    index/cpu_index_mirror.h index/cpu_index_mirror.cpp
    index/cpu_dfs.h index/cpu_dfs.cpp
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_cpu_dfs PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_cpu_dfs PRIVATE
    USE_GAMMA
    USE_VALID_BITS
    USE_MERGED_MATCHING
    ENABLE_CPU_DFS
)

target_compile_options(cgcsm_cpu_dfs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_cpu_dfs PUBLIC
    link_cpu_dfs
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)
add_executable(graph_sampler
    sampler/graph.h sampler/graph.cpp
    sampler/graph_sampler.h
    sampler/main.cpp)

target_compile_options(graph_sampler PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -O3 -march=native
    >
)

# Motivation experiment: standalone CaLiG index maintenance with tryNei counting
add_executable(motivation
    index/calig.cpp
    main_motivation.cpp)

target_compile_options(motivation PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -O3 -march=native
    >
)

target_compile_definitions(motivation PRIVATE
    COUNT_TRYNEI
)

target_link_libraries(motivation PUBLIC
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

add_executable(cgcsm_merged_matching_global
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_merged_matching_global PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_merged_matching_global PRIVATE
    USE_VALID_BITS
    USE_MERGED_MATCHING
    USE_GLOBAL_RQ
)

target_compile_options(cgcsm_merged_matching_global PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_merged_matching_global PUBLIC
    link_global
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)

# Default order variant: forces matching order to start from update edge
add_executable(cgcsm_default_order
    graph/graph.h graph/graph.cpp
    graph/plan.h graph/plan.cpp
    index/calig.h index/calig.cpp
    index/calig_helper_gamma.h index/calig_helper_gamma.cu
    utils/config.h utils/hash_utils.h
    utils/constants.h utils/constants.cpp
    utils/types.h
    main_calig.cpp)

set_target_properties(cgcsm_default_order PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    LINKER_LANGUAGE CXX
)

target_compile_definitions(cgcsm_default_order PRIVATE
    FORCE_UPDATE_EDGE_START
)

target_compile_options(cgcsm_default_order PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:-G -g>
        $<$<CONFIG:Release>:
            -O3
            -lineinfo
            --ptxas-options=-v
        >
        -Xcompiler -march=native
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wno-unused-command-line-argument
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:
            -O3
        >
    >
)

target_link_libraries(cgcsm_default_order PUBLIC
    link
    nd
    -lcudart
    TBB::tbb
    TBB::tbbmalloc
    TBB::tbbmalloc_proxy
    -lnuma
)
