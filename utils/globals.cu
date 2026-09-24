
#include <cstdint>

#include "utils/config.h"
#include "utils/mem_pool.h"
#include "utils/globals.h"

__constant__ uint32_t C_QE_COUNT;
__constant__ uint32_t C_QV_COUNT;
__constant__ uint32_t C_DV_COUNT;
__constant__ uint8_t C_NLF[MAX_ECOUNT * 2];
__constant__ uint8_t C_QV_OFFS[MAX_VCOUNT + 1];

__constant__ uint8_t C_EIDX[MAX_VCOUNT * MAX_VCOUNT];

__constant__ CyclicQueue<uint32_t> C_RES_QUEUE;
