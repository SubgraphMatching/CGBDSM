#ifndef SYMBI_GRAPH_GRAPH
#define SYMBI_GRAPH_GRAPH

#include <queue>
#include <tuple>
#include <vector>
#include <climits>
#include <bits/stdc++.h>

#define NOT_EXIST UINT_MAX
#define UNMATCHED UINT_MAX

struct InsertUnit {
    char type;  // 'v' or 'e' 
    bool is_add;// addition or deletion
    uint id1;   // vertex id or edge source id
    uint id2;   // edge target id
    uint label; // vertex or edge label
    InsertUnit(char type_arg, bool is_add_arg, uint id1_arg, uint id2_arg, uint label_arg)
    : type(type_arg), is_add(is_add_arg), id1(id1_arg), id2(id2_arg), label(label_arg) {}
};

class Graph
{
protected:
    uint edge_count_;
    uint vlabel_count_;
    uint elabel_count_;
    std::vector<std::vector<uint>> neighbors_;
    std::vector<std::vector<uint>> elabels_;

public:
    std::vector<InsertUnit> updates_;
    std::vector<uint> vlabels_;

public:
    Graph();

    uint NumVertices() const { return vlabels_.size(); }
    uint NumEdges() const { return edge_count_; }
    uint NumVLabels() const { return vlabel_count_; }
    uint NumELabels() const { return elabel_count_; }

    void AddVertex(uint id, uint label);
    void RemoveVertex(uint id);
    void AddEdge(uint v1, uint v2, uint label);
    void RemoveEdge(uint v1, uint v2);

    uint GetVertexLabel(uint u) const;
    const std::vector<uint>& GetNeighbors(uint v) const;
    const std::vector<uint>& GetNeighborLabels(uint v) const;
    uint GetDegree(uint v) const;
    std::tuple<uint, uint, uint> GetEdgeLabel(uint v1, uint v2) const;

    void LoadFromFile(const std::string &path);
    void LoadUpdateStream(const std::string &path);
    void PrintMetaData() const;
};

#endif //SYMBI_GRAPH_GRAPH
