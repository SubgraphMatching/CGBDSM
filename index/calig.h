#ifndef CALIG_H
#define CALIG_H

#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include "index/storage_hash_map.hpp"

// class QueryGraph;

using u_set = ska::flat_hash_set<int>;
using vec = std::vector<int>;

struct vertex_Q {
    int label;
    u_set nei;
    ska::flat_hash_map<int, vec> rep_nei;
};

struct vertex_G {
    int label;
    u_set nei;
    ska::flat_hash_map<int, ska::flat_hash_map<int, u_set>> cand;
    ska::flat_hash_map<int, bool> isChecked;
};

struct DeltaOp { uint32_t vi, vj, ui, uj; };

struct Task { int ui; int vi; };
struct DelTask { int ui; int vi; int lb; };
struct TmpTask { int ui; int uj; int vi; int vj; };
using CandMap = ska::flat_hash_map<int, u_set>;

class CaLiG {
public:
    CaLiG();
    ~CaLiG();

    void inputQ(std::string &qp);
    void inputGBin(const std::string& bin_filename);
    void inputG(std::string &gp);
    void inputUpdate(std::string &path, int max_num);

    void constructCand(uint32_t num_batch);
    void staticFilter();
    void updateIndex(uint32_t start, uint32_t count, uint32_t batch);
    void ConstructUpdate(uint32_t start, uint32_t count);

    // void DetectConflict(uint32_t start, uint32_t count);
    // void setQueryGraph(const QueryGraph& qg) { dc_qg_ = &qg; }
    // const std::vector<uint8_t>& getEdgeConflict() const { return edge_conflict_; }
    // uint32_t getDCStart() const { return dc_start_; }
    // uint32_t getDCNumPairs() const { return dc_num_pairs_; }
    // const vec& getUpdate() const { return update; }

    // uint64_t dc_reached_ = 0, dc_conflicted_ = 0;
    uint32_t getUpdateSize() const { return update.size(); }
    void count_candidate();

#ifdef COUNT_TRYNEI
    std::vector<uint64_t> tryNei_count_;
    uint64_t getTryNeiTotal() const {
        uint64_t sum = 0;
        for (auto& c : tryNei_count_) sum += c;
        return sum;
    }
    void resetTryNeiCount() {
        for (auto& c : tryNei_count_) c = 0;
    }
#endif

    std::vector<vertex_Q> Q;
    std::vector<vertex_G> G;
    std::vector<uint32_t> G_Li;  // Standalone LI bitmask array for cache locality
    std::vector<uint32_t> li_changed_global_;

    // Flat delta vectors (replace G_UPDATE/G_UPDATED)
    std::vector<std::vector<DeltaOp>> delta_ins_local_;
    std::vector<std::vector<DeltaOp>> delta_del_local_;
    std::vector<std::vector<DeltaOp>> delta_update_local_;
    std::vector<std::vector<DeltaOp>> delta_ins_merged_;
    std::vector<std::vector<DeltaOp>> delta_del_merged_;
    std::vector<std::vector<DeltaOp>> delta_update_merged_;

    const std::vector<DeltaOp>& getInsDeltas(uint32_t batch) const { return delta_ins_merged_[batch]; }
    const std::vector<DeltaOp>& getDelDeltas(uint32_t batch) const { return delta_del_merged_[batch]; }
    const std::vector<DeltaOp>& getUpdateDeltas(uint32_t batch) const { return delta_update_merged_[batch]; }
    void mergeDeltas(uint32_t batch);

    void setNumThreads(int n) { num_threads_ = n; }
    int getNumThreads() const { return num_threads_; }

private:
    int Q_size = 0, G_size = 0, batch = 0;
    vec update;
    ska::flat_hash_map<int, vec> labels;
    int num_threads_ = 1;

    // const QueryGraph* dc_qg_ = nullptr;
    // std::vector<uint32_t> dc_ei_of_;     // [u*qsize+uu] -> undirected ei (from QueryGraph::qe_list_)
    // uint32_t dc_ecount_ = 0;             // undirected query-edge count (= QE_COUNT)
    // uint32_t dc_start_ = 0;              // batch window base into `update`
    // uint32_t dc_num_pairs_ = 0;          // batch window size (in edge pairs)
    // std::vector<uint32_t> dc_owner_;     // owner3d[ei][qv][dv] (generation-tagged, reused, no clear)
    // uint64_t dc_tag_counter_ = 1;        // generation base for dc_owner_ tags
    // std::vector<uint8_t> edge_conflict_; // [ei*num_pairs + k]: 1 if update edge k conflicts on ei

    static constexpr int LOCK_COUNT = 163840;
    std::mutex cond_locks[LOCK_COUNT];

    std::vector<std::vector<TmpTask>> local_tmp_;
    std::vector<TmpTask> global_tmp_;
    std::vector<std::vector<TmpTask>> bucket_;
    std::vector<std::vector<Task>> local_check_;
    std::vector<Task> global_check_;
    std::vector<std::vector<DelTask>> local_check_del_;
    std::vector<DelTask> global_check_del_;
    std::vector<std::vector<DelTask>> update_queue;
    std::vector<DelTask> global_update_queue;
    std::vector<std::vector<Task>> add_queue, del_queue, add_del_check_queue, check_queue;
    std::vector<Task> global_add_queue, global_del_queue, global_add_del_check_queue, global_check_queue;
    std::vector<std::vector<int>> del_check_queue;
    std::vector<int> global_del_check_queue;
    std::vector<std::vector<TmpTask>> tmp_queue, tmp_queue_1;
    std::vector<TmpTask> global_tmp_queue, global_tmp_queue_1;
    std::vector<std::vector<Task>> li_activated_queue;
    std::vector<Task> global_li_activated;
    std::vector<std::vector<uint32_t>> li_changed_local_;

    inline int getLockId(int v, int ui, int uj);
    void lockVertex(int v, int ui = 0, int uj = 0);
    void unlockVertex(int v, int ui = 0, int uj = 0);

    template <typename T>
    void mergeTasks(std::vector<std::vector<T>> &local, std::vector<T> &global);

    bool tryNei(int th, int vi, int ui, const CandMap& cand_ui, u_set &used, vec &to_check);
    bool checkNei(int vi, int ui);

    void delAndCheck(std::vector<Task> &global, std::vector<std::vector<Task>> &local);
    void turnOff(int &vi, std::vector<std::vector<Task>> &local_del);
    void turnOffCheck(std::vector<int> &global_check, std::vector<TmpTask> &global_tmp, std::vector<std::vector<Task>> &local_del);

    void turnOffCond(int v1, int v2, std::vector<std::vector<int>> &local_check, std::vector<std::vector<TmpTask>> local_tmp);

    void addAndCheck(std::vector<Task> &global, std::vector<std::vector<Task>> &local, std::vector<std::vector<Task>> &local_add_del);
    void addAndCheckDelete(std::vector<Task> &global_check_del, std::vector<std::vector<Task>> &local_del);

    void turnOnCheck(std::vector<TmpTask> &global_tmp, std::vector<std::vector<Task>> &local_add);
    void turnOnCond(int &v1, int &v2, std::vector<std::vector<Task>> &local, std::vector<std::vector<TmpTask>> &local_tmp);
};
#endif // CALIG_H
