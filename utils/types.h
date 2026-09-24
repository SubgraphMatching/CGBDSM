#ifndef UTILS_TYPES_H
#define UTILS_TYPES_H

#include <cstdint>
#include <vector>

struct Tries
{
    uint32_t vs_size_;
    uint32_t es_size_;
    uint32_t *vs_;
    uint32_t *offs_;
    uint32_t *nbrs_;
};

struct TrieCapability
{
    uint32_t vs_capability_;
    uint32_t off_capability_;
    uint32_t es_capability_;
};

using EdgeList = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

using EdgeBatch = std::vector<EdgeList>;

#endif