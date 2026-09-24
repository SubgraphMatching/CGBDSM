#include "graph.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>

Graph loadGraph(const std::string& path) {
    if (path.size() >= 6 && path.compare(path.size() - 6, 6, ".graph") == 0)
        return loadGraphText(path);
    else
        return loadGraphBinary(path);
}

Graph loadGraphText(const std::string& path) {
    Graph g;
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Cannot open graph file: " << path << "\n";
        return g;
    }

    char c;
    int id, id1, id2, lb;

    while (fin >> c) {
        if (c == 'v') {
            fin >> id >> lb;
            Vertex v;
            v.label = lb;
            for (int i = g.size; i < id; i++)
                g.vertices.push_back({-1});
            g.vertices.push_back(v);
            g.size = g.vertices.size();
        } else {
            fin >> id1 >> id2 >> lb;
            if (id1 < g.size && id2 < g.size
                && g.vertices[id1].label != -1 && g.vertices[id2].label != -1) {
                g.vertices[id1].nei.insert(id2);
                g.vertices[id2].nei.insert(id1);
            }
            break;
        }
    }
    while (fin >> c) {
        fin >> id1 >> id2 >> lb;
        if (id1 < g.size && id2 < g.size
            && g.vertices[id1].label != -1 && g.vertices[id2].label != -1) {
            g.vertices[id1].nei.insert(id2);
            g.vertices[id2].nei.insert(id1);
        }
    }
    return g;
}

Graph loadGraphBinary(const std::string& prefix) {
    Graph g;
    std::string meta_path   = prefix + ".meta.txt";
    std::string vertex_path = prefix + ".vertex.bin";
    std::string edge_path   = prefix + ".edge.bin";

    std::ifstream meta_in(meta_path);
    if (!meta_in) {
        std::cerr << "Cannot open meta file: " << meta_path << "\n";
        return g;
    }

    uint64_t n_edges = 0;
    uint32_t n_vertices = 0, vsize = 0, esize = 0;
    uint32_t vlbl_sz = 0, elbl_sz = 0, meta_max_deg = 0;
    uint32_t feat = 0, nvcls = 0, necls = 0;

    meta_in >> n_vertices >> n_edges
            >> vsize >> esize >> vlbl_sz >> elbl_sz
            >> meta_max_deg >> feat >> nvcls >> necls;
    meta_in.close();

    if (vsize != 4 || esize != 8) {
        std::cerr << "Unsupported id size (expected vertex:4 byte, edge:8 byte)\n";
        return g;
    }

    std::vector<uint64_t> row_ptr(n_vertices + 1);
    std::ifstream v_in(vertex_path, std::ios::binary);
    if (!v_in.good()) {
        std::cerr << "Failed to open " << vertex_path << "\n";
        return g;
    }
    v_in.read(reinterpret_cast<char*>(row_ptr.data()), sizeof(uint64_t) * (n_vertices + 1));

    std::vector<uint32_t> edges(n_edges);
    std::ifstream e_in(edge_path, std::ios::binary);
    if (!e_in.good()) {
        std::cerr << "Failed to open " << edge_path << "\n";
        return g;
    }
    e_in.read(reinterpret_cast<char*>(edges.data()), sizeof(uint32_t) * n_edges);

    g.vertices.resize(n_vertices);
    g.size = n_vertices;

    for (uint32_t v = 0; v < n_vertices; v++) {
        g.vertices[v].label = rand() % nvcls;
    }

    for (uint32_t v = 0; v < n_vertices; v++) {
        for (uint64_t i = row_ptr[v]; i < row_ptr[v + 1]; i++) {
            uint32_t u = edges[i];
            if (u < n_vertices) {
                g.vertices[v].nei.insert(u);
                g.vertices[u].nei.insert(v);
            }
        }
    }

    std::cout << "Binary graph loaded: " << n_vertices << " vertices, "
              << n_edges << " directed edges\n";
    return g;
}
