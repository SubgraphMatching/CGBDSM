#ifndef SAMPLER_GRAPH_H
#define SAMPLER_GRAPH_H

#include <vector>
#include <unordered_set>
#include <string>

struct Vertex {
    int label;
    std::unordered_set<int> nei;
};

struct Graph {
    std::vector<Vertex> vertices;
    int size = 0;
};

Graph loadGraph(const std::string& path);
Graph loadGraphText(const std::string& path);
Graph loadGraphBinary(const std::string& prefix);

#endif
