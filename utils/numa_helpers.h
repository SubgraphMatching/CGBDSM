#ifndef UTILS_NUMA_HELPERS_H
#define UTILS_NUMA_HELPERS_H

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <vector>

// Try to use libnuma if available, fall back to single-node assumption
#if defined(__has_include)
  #if __has_include(<numa.h>)
    #define HAS_NUMA_H 1
    #include <numa.h>
  #endif
#endif

// Returns the number of available NUMA nodes
inline int getNumNumaNodes() {
#ifdef HAS_NUMA_H
    if (numa_available() >= 0) {
        int num_nodes = numa_num_configured_nodes();
        return num_nodes > 0 ? num_nodes : 1;
    }
#endif
    return 1;
}

// Returns the NUMA node of the current thread
inline int getCurrentNumaNode() {
#ifdef HAS_NUMA_H
    if (numa_available() >= 0) {
        return numa_preferred();
    }
#endif
    return 0;
}

// Returns which NUMA node a vertex belongs to (by modulo partitioning)
inline int getVertexNumaNode(uint32_t vertex_id, int num_nodes) {
    return vertex_id % num_nodes;
}

// Bind the current thread to the specified NUMA node's CPUs
inline void bindThreadToNumaNode(int node) {
#ifdef HAS_NUMA_H
    if (numa_available() < 0 || numa_num_configured_nodes() <= 1) {
        return;
    }

    struct bitmask *cpumask = numa_allocate_cpumask();

    struct bitmask *node_cpus = numa_allocate_cpumask();
    numa_node_to_cpus(node, node_cpus);

    for (size_t i = 0; i < node_cpus->size; i++) {
        if (numa_bitmask_isbitset(node_cpus, i)) {
            numa_bitmask_setbit(cpumask, i);
        }
    }

    numa_free_cpumask(node_cpus);

    numa_bind(cpumask);
    numa_set_preferred(node);

    numa_free_cpumask(cpumask);
#else
    (void)node;
#endif
}

// Initialize NUMA awareness (call at program start)
inline void initNumaAware(int* num_nodes_out = nullptr) {
    int num_nodes = getNumNumaNodes();
    if (num_nodes_out) *num_nodes_out = num_nodes;

    std::cout << "[NUMA] Detected " << num_nodes << " NUMA node(s)" << std::endl;
    if (num_nodes > 1) {
        std::cout << "[NUMA] NUMA-aware optimization enabled" << std::endl;
    } else {
        std::cout << "[NUMA] Single NUMA node, optimization disabled" << std::endl;
    }
}

#endif // UTILS_NUMA_HELPERS_H
