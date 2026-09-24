#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <tbb/task_arena.h>
#include "utils/numa_helpers.h"
#include "index/calig.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <query_path> <data_path> <update_path>\n";
        return 1;
    }

    std::string query_path = argv[1];
    std::string data_path  = argv[2];
    std::string update_path = argv[3];

    int num_numa_nodes = getNumNumaNodes();
    int total_cores = std::thread::hardware_concurrency();
    int cores_per_node = total_cores / std::max(num_numa_nodes, 1);
    int s1_numa = 0;

    auto s1_c = tbb::task_arena::constraints{}
        .set_numa_id(s1_numa)
        .set_max_concurrency(cores_per_node);
    tbb::task_arena s1_arena(s1_c);

    s1_arena.execute([&]() {
        CaLiG calig;
        calig.inputQ(query_path);
        if (data_path.compare(data_path.length() - 6, 6, ".graph") == 0) {
            calig.inputG(data_path);
            calig.inputUpdate(update_path, 0);
        } else {
            calig.inputGBin(data_path);
        }

        uint32_t num_updates = calig.getUpdateSize();
        std::cout << "Updates: " << num_updates << "\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        calig.constructCand(1);
        calig.staticFilter();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Init (constructCand + staticFilter): "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

        t0 = std::chrono::high_resolution_clock::now();
#ifdef COUNT_TRYNEI
        calig.resetTryNeiCount();
#endif
        calig.updateIndex(0, num_updates, 0);
        t1 = std::chrono::high_resolution_clock::now();
        double update_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "Update time: " << update_ms << " ms\n";
#ifdef COUNT_TRYNEI
        std::cout << "check count: " << calig.getTryNeiTotal() << "\n";
#endif
    });

    return 0;
}
