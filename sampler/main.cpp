#include <iostream>
#include <string>
#include <sys/stat.h>
#include "graph.h"
#include "graph_sampler.h"

static void mkdirp(const std::string& path) {
    for (size_t i = 1; i < path.size(); i++) {
        if (path[i] == '/')
            mkdir(path.substr(0, i).c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_path> <output_dir> <sample_size> <num_samples> <type> [seed]\n"
                  << "  type: dense | sparse | tree | all\n";
        return 1;
    }

    std::string data_path = argv[1];
    std::string output_dir = argv[2];
    int sample_size = std::stoi(argv[3]);
    int num_samples = std::stoi(argv[4]);
    std::string type_str = argv[5];
    uint64_t seed = (argc > 6) ? std::stoull(argv[6]) : 0;

    std::cout << "Loading graph from " << data_path << " ...\n";
    Graph graph = loadGraph(data_path);

    int valid_count = 0;
    for (int v = 0; v < graph.size; v++)
        if (graph.vertices[v].label != -1) valid_count++;
    std::cout << "Loaded: " << graph.size << " vertices ("
              << valid_count << " valid)\n";

    // 构建 base 目录: <output_dir>/<size>_self/<type>/
    std::string size_dir = output_dir + "/" + std::to_string(sample_size) + "_self";

    auto runType = [&](QueryType type, const std::string& name) {
        std::string type_dir = size_dir + "/" + name;
        mkdirp(type_dir);

        GraphSampler sampler(graph, seed);
        auto subgraphs = sampler.sampleMultiple(sample_size, num_samples, type);

        int connected_count = 0;
        for (int i = 0; i < (int)subgraphs.size(); i++) {
            std::string path = type_dir + "/Q_" + std::to_string(i);
            GraphSampler::writeToFile(subgraphs[i], path);
            double davg = GraphSampler::avgDegree(subgraphs[i]);
            bool connected = GraphSampler::isConnected(subgraphs[i]);
            if (connected) connected_count++;
            std::cout << name << " Q_" << i << ": "
                      << subgraphs[i].vertices.size() << "v "
                      << subgraphs[i].edges.size() << "e "
                      << "davg=" << davg
                      << (connected ? "" : " [DISCONNECTED!]")
                      << "\n";
        }
        std::cout << name << ": " << subgraphs.size() << "/" << num_samples
                  << " sampled, " << connected_count << " connected\n\n";
    };

    if (type_str == "all") {
        runType(QueryType::Tree, "tree");
        runType(QueryType::Dense, "dense");
        runType(QueryType::Sparse, "sparse");
    } else if (type_str == "dense") {
        runType(QueryType::Dense, "dense");
    } else if (type_str == "sparse") {
        runType(QueryType::Sparse, "sparse");
    } else if (type_str == "tree") {
        runType(QueryType::Tree, "tree");
    } else {
        std::cerr << "Unknown type: " << type_str << " (use dense/sparse/tree/all)\n";
        return 1;
    }

    return 0;
}
