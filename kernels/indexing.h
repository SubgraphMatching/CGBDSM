#ifndef KERNELS_INDEXING
#define KERNELS_INDEXING

#include <cstdint>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "graph/graph_gpu.h"

__global__ void setCapabilities(
    const Tries tries,
    uint32_t *capability
);

__global__ void roundCapabilities(
    uint32_t *array,
    const uint32_t size
);

__global__ void roundCapabilities(
    uint32_t *array,
    const uint32_t size,
    const uint32_t *label_array,
    const uint32_t q_label
);

__global__ void setNeighborPointers(
    uint32_t *base,
    const uint32_t *offsets,
    const uint32_t size,
    uint32_t **ptr
);

__global__ void allocateFromMemPool(
    RelationsGPU data, const uint32_t idx,
    const uint32_t num_vertices, MemPool<uint32_t> pool
);

__global__ void removeTriesFromGraph(
    const Tries del_tries,
    RelationsGPU data,
    const uint32_t idx
);

__global__ void addTriesToGraph(
    const Tries tries,
    RelationsGPU data,
    const uint32_t idx,
    MemPool<uint32_t> nbr_mem_pool
);

// Direct CSR merge kernels (no Tries intermediate)
__global__ void setCapabilitiesFromSizes(
    uint32_t *sizes, uint32_t *capability, uint32_t n);

__global__ void copyFromFlatCSR(
    uint32_t **old_nbrs, uint32_t *sizes,
    uint32_t **new_nbrs, uint32_t num_vertices);

__global__ void mergeCSROntoGraph(
    RelationsGPU src,
    RelationsGPU dst,
    const uint32_t idx,
    MemPool<uint32_t> nbr_mem_pool,
    uint32_t num_vertices);

__global__ void removeCSROromGraph(
    RelationsGPU src,
    RelationsGPU dst,
    const uint32_t idx,
    uint32_t num_vertices);

__global__ void statisticIndex(
    const RelationsGPU data,
    const uint32_t idx,
    uint32_t *sum,
    uint32_t *count
);

__global__ void getGlobalCandidates(
    const RelationsGPU data,
    const uint32_t idx,
    const Tries tries,
    uint32_t *cand_bits,
    bool *cand_flag,
    const uint32_t u
);

__global__ void getGlobalCandidateEdgesCount(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count
);

__global__ void getGlobalCandidateEdgesWrite(
    const RelationsGPU data,
    const uint32_t idx,
    const uint32_t *new_cand,
    const uint32_t new_cand_size,
    const uint32_t *other_cand_bits,
    uint32_t *cand_e_count_prefix_sum,
    uint32_t *relation_u,
    uint32_t *relation_uu
);

__global__ void filterRelevantCount(
    const Tries input,
    Tries output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void filterRelevantWrite(
    const Tries input,
    Tries output,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void edgeList2RelationCount(
    const Tries input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void edgeList2RelationWrite(
    const Tries input,
    RelationsGPU output,
    const uint8_t idx,
    const uint32_t *first_candidates,
    const uint32_t *second_candidates
);

__global__ void getLocalCandidatesBackward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void getLocalCandidatesForward(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t forward_index,
    const uint8_t backward_index,
    const uint8_t first_bn_of_uu_index,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void buildLocalRelationNew2OldCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void buildLocalRelationNew2OldWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
);

__global__ void buildLocalRelationOld2NewCount(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void buildLocalRelationOld2NewWrite(
    const RelationsGPU global_index,
    RelationsGPU local_index,
    const uint8_t idx,
    const uint8_t first_bn_of_uu_index,
    const uint32_t *cum_bn,
    const uint32_t pre_max_cum,
    uint32_t *relation_u
);

__global__ void mapTrieToRelation(
    RelationsGPU local_index,
    const uint8_t idx,
    const uint32_t *vs,
    const uint32_t *offs,
    const uint32_t num_items
);

// Valid bit kernels for BuildLocalIndexBit
__global__ void setLocalCandidateValidBits(
    const uint32_t *cum_bn,
    const uint32_t max_cum,
    const uint8_t qv_idx
);

__global__ void setInitialValidBits(
    const RelationsGPU index,
    const uint8_t idx,
    const uint8_t qv_idx
);

__global__ void getLocalCandidatesBackwardBit(
    const RelationsGPU global_index,
    const uint8_t backward_index,
    const uint8_t bn_qv,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

__global__ void getLocalCandidatesForwardBit(
    const RelationsGPU global_index,
    const uint8_t forward_index,
    const uint8_t backward_index,
    const uint8_t bn_qv,
    uint32_t *cum_bn,
    const uint32_t pre_max_cum
);

// Merged valid bit kernels (per-edge, dual-source with visibility masking)
__global__ void setInitialValidBitsAll(
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs
);

__global__ void checkAllConstraintsAndSetBitsAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
);

__global__ void pushFromFirstBNAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
);

__global__ void verifyCandidatesAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
);

__global__ void pushAllBackwardBNAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs,
    smask_t** d_edge_sm_tmp_ptrs
);

__global__ void intersectAndSetBitsAll(
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs,
    smask_t** d_edge_sm_tmp_ptrs
);

__global__ void forwardLookaheadAll(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
);

__global__ void checkEdgeEmptyAll(
    const RelationsGPU update_index,
    uint8_t* d_flags
);

__global__ void shallowDFSExpand(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs,
    const uint8_t end_depth
);

__global__ void writeInitialShallowDFS(
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs,
    unsigned long long int new_res,
    unsigned long long int *new_res_size,
    const unsigned long long int h_max_new_res_size
);

__global__ void shallowDFSExpandFromQueue(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs,
    unsigned long long int res,
    const unsigned long long int res_size,
    const uint8_t end_depth
);

// 1-bit packed support mask kernels (SUPPORT_MASK_WIDTH == 1)
#if SUPPORT_MASK_WIDTH == 1
__global__ void setInitialValidBitsAll1Bit(
    const RelationsGPU update_index,
    smask_t** d_edge_sm_ptrs
);

__global__ void checkAllConstraintsAndSetBitsAll1Bit(
    const RelationsGPU index_gpu,
    const RelationsGPU update_index,
    const uint8_t depth,
    smask_t** d_edge_sm_ptrs
);
#endif

#endif
