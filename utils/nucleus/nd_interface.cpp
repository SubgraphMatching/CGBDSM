//
// Created by ssunah on 3/10/20.
//
#include <cstring>
#include "nd_interface.h"

void nd_interface::nd(nd_Graph& graph, uint32_t r, uint32_t s, vector<nd_tree_node> &nd_tree) {
    int nEdge = 0;

    for (uint32_t i = 0; i < graph.size(); i++)
    {
        nEdge += graph[i].size();
    }
    nEdge /= 2;

    int maxK; // maximum K value in the graph
    vector<int> K;

    if (r == 1 && s == 2) {
        base_kcore(graph, nEdge, K, &maxK, nd_tree);
    }
    else if (r == 2 && s == 3) {
        base_ktruss(graph, nEdge, K, &maxK, nd_tree);
    }
    else if (r == 3 && s == 4) {
        base_k34(graph, nEdge, K, &maxK, nd_tree);
    }
    else {
        std::cout << "The nucleus decomposition with r = " << r << " and s = " << s << " is not supported." << std::endl;
        exit(-1);
    }
}

void nd_interface::print_nd_tree(uint32_t r, uint32_t s, std::vector<nd_tree_node> &nd_tree) {
    std::vector<int> forest;
    for (auto& node : nd_tree) {
        if (node.parent_ == -1) {
            forest.push_back(node.id_);
        }
    }

    printf("The number of %d-%d trees: %zu\n", r, s, forest.size());

    for (auto root : forest) {
        std::queue<int> q;
        q.push(root);
        printf ("\tTree:\n");
        while (!q.empty()) {
            int node_id = q.front();
            q.pop();
            printf ("\t\tid %d, parent %d, k %d, |V| %zu, |E| %d, density %.4f, num of children %zu: ",
                    nd_tree[node_id].id_, nd_tree[node_id].parent_, nd_tree[node_id].k_, nd_tree[node_id].vertices_.size(),
                    nd_tree[node_id].num_edges_, nd_tree[node_id].density_, nd_tree[node_id].children_.size());
            for (auto u : nd_tree[node_id].vertices_) {
                printf ("%d ", u);
            }
            printf("\n");

            for (auto child : nd_tree[node_id].children_) {
                q.push(child);
            }
        }
    }

}
