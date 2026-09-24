#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <algorithm>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include <tbb/concurrent_vector.h>
#include <atomic>
#include "utils/numa_helpers.h"
#include "index/calig.h"
#include "graph/graph.h"

using namespace std;

inline int CaLiG::getLockId(int v, int ui, int uj)
{
    unsigned int x = (unsigned int)v;
    x ^= (unsigned int)(ui << 21);
    x ^= (unsigned int)(uj << 26);
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return (int)(x % LOCK_COUNT);
}

inline void CaLiG::lockVertex(int v, int ui, int uj)
{
    cond_locks[getLockId(v, ui, uj)].lock();
}

inline void CaLiG::unlockVertex(int v, int ui, int uj)
{
    cond_locks[getLockId(v, ui, uj)].unlock();
}

double get_ms(std::chrono::high_resolution_clock::time_point start)
{
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
}

template <class ET>
inline bool CAS(ET *ptr, ET oldv, ET newv)
{
    if (sizeof(ET) == 1)
        return __sync_bool_compare_and_swap((bool *)ptr, *((bool *)&oldv),
                                            *((bool *)&newv));
    else if (sizeof(ET) == 4)
        return __sync_bool_compare_and_swap((int *)ptr, *((int *)&oldv),
                                            *((int *)&newv));
    else if (sizeof(ET) == 8)
        return __sync_bool_compare_and_swap((long *)ptr, *((long *)&oldv),
                                            *((long *)&newv));
    std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
    abort();
}

template <class ET>
inline ET writeAnd(ET *a, ET b)
{
    ET oldV, newV;
    do
    {
        oldV = *a;
        newV = oldV & b;
    } while (!CAS(a, oldV, newV));
    return oldV;
}

template <class ET>
inline ET writeOr(ET *a, ET b)
{
    ET oldV, newV;
    do
    {
        oldV = *a;
        newV = oldV | b;
    } while (!CAS(a, oldV, newV));
    return oldV;
}

template <typename T>
void CaLiG::mergeTasks(vector<vector<T>> &local, vector<T> &global)
{
    int num_threads = local.size();
    std::vector<size_t> offsets(num_threads + 1, 0);
    size_t total = 0;
    for (int i = 0; i < num_threads; ++i) {
        offsets[i] = total;
        total += local[i].size();
    }
    offsets[num_threads] = total;
    global.resize(total);

    tbb::parallel_for(0, num_threads, [&](int i) {
        if (!local[i].empty())
            memcpy(global.data() + offsets[i], local[i].data(), local[i].size() * sizeof(T));
        local[i].clear();
    });
}

void CaLiG::inputQ(string &qp)
{
    ifstream qg(qp);
    if (!qg)
        cerr << "Fail to open query file." << endl;

    char c;
    int id, id1, id2, lb, dvir;
    vertex_Q u;
    while (qg >> c)
    {
        if (c == 'v')
        {
            qg >> id >> lb;
            if (lb == -1)
                lb = 0;
            u.label = lb;
            Q.push_back(u);
        }
        else
        {
            qg >> id1 >> id2 >> lb;
            Q[id1].nei.insert(id2);
            Q[id2].nei.insert(id1);
            break;
        }
    }
    while (qg >> c)
    {
        qg >> id1 >> id2 >> lb;
        Q[id1].nei.insert(id2);
        Q[id2].nei.insert(id1);
    }
    qg.close();
    Q_size = Q.size();

    for (int ui = 0; ui < Q_size; ui++)
    {
        labels[Q[ui].label].emplace_back(ui);
        ska::flat_hash_map<int, vec> rep_nei;
        for (auto &uni : Q[ui].nei)
        {
            int &uni_label = Q[uni].label;
            rep_nei[uni_label].emplace_back(uni);
        }
        for (auto &repi : rep_nei)
        {
            if (repi.second.size() > 1)
            {
                Q[ui].rep_nei[repi.first] = repi.second;
            }
        }
    }
}

void CaLiG::inputG(string &gp)
{
    ifstream dg(gp);
    if (!dg)
        cerr << "Fail to open graph file." << endl;

    char c;
    int id, id1, id2, lb;
    vertex_G v, tmp;
    while (dg >> c)
    {
        if (c == 'v')
        {
            dg >> id >> lb;
            if (labels.count(lb))
                v.label = lb;
            else
                v.label = -1;
            for (int i = G.size(); i < id; i++)
            {
                tmp.label = -1;
                G.push_back(tmp);
            }
            if (id != G.size())
                cerr << id << " " << G.size() << endl;
            G.push_back(v);
        }
        else
        {
            G_size = G.size();
            G_Li.resize(G_size, 0);
            dg >> id1 >> id2 >> lb;
            if (id1 < G_size && id2 < G_size && G[id1].label != -1 && G[id2].label != -1)
            {
                G[id1].nei.insert(id2);
                G[id2].nei.insert(id1);
            }
            break;
        }
    }
    while (dg >> c)
    {
        dg >> id1 >> id2 >> lb;
        if (id1 < G_size && id2 < G_size && G[id1].label != -1 && G[id2].label != -1)
        {
            G[id1].nei.insert(id2);
            G[id2].nei.insert(id1);
        }
    }
    dg.close();
}

void CaLiG::inputGBin(const std::string& prefix) {
    std::string meta_path   = prefix + ".meta.txt";
    std::string vertex_path = prefix + ".vertex.bin";
    std::string edge_path   = prefix + ".edge.bin";

    std::ifstream meta_in(meta_path);
    if (!meta_in) {
        std::cerr << "Cannot open meta file: " << meta_path << "\n";
        return;
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
        return;
    }

    std::vector<uint64_t> row_ptr(n_vertices + 1);
    std::ifstream v_in(vertex_path, std::ios::binary);
    if (!v_in.good()) {
        std::cerr << "Failed to open " << vertex_path << "\n";
        return;
    }
    v_in.read(reinterpret_cast<char*>(row_ptr.data()), sizeof(uint64_t) * (n_vertices + 1));

    if (row_ptr.back() != n_edges) {
        std::cerr << "CSR inconsistency: row_ptr[" << n_vertices << "] = " 
                  << row_ptr.back() << " != n_edges " << n_edges << "\n";
    }

    std::vector<uint32_t> edges(n_edges);
    std::ifstream e_in(edge_path, std::ios::binary);
    if (!e_in.good()) {
        std::cerr << "Failed to open " << edge_path << "\n";
        return;
    }
    e_in.read(reinterpret_cast<char*>(edges.data()), sizeof(uint32_t) * n_edges);

    G.resize(n_vertices);
    G_size = n_vertices;
    G_Li.resize(G_size, 0);
    for (auto& v : G) {
        int lb = rand() % nvcls;
        if (labels.count(lb))
            v.label = lb;
        else
            v.label = -1;
    }

    auto should_be_update = [](uint32_t a, uint32_t b) -> bool {
        uint32_t mn = std::min(a, b), mx = std::max(a, b);
        uint64_t key = ((uint64_t)mn << 32) | mx;
        uint64_t hash_val = key * 0x9e3779b97f4a7c15ULL;
        return (hash_val % 10ULL) == 0;
    };

    std::vector<std::vector<int>> local_update;
    local_update.resize(num_threads_);

    for(int v = 0; v < n_vertices; v++) {
        uint64_t start = row_ptr[v], end = row_ptr[v + 1];
        auto& neighbors_set = G[v].nei;
        for (uint64_t i = start; i < end; ++i) {
            uint32_t u = edges[i];
            bool is_update_edge = should_be_update(v, u);
            if(is_update_edge) {
                if(v < u) {
                    update.push_back(u), update.push_back(v);
                }
            }
            else {
                neighbors_set.insert(u);
            }
        }
    }

    // Deterministic shuffle of update edge pairs
    {
        int n_pairs = update.size() / 2;
        std::vector<int> idx(n_pairs);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(42);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<int> shuffled(update.size());
        for (int i = 0; i < n_pairs; i++) {
            shuffled[i * 2]     = update[idx[i] * 2];
            shuffled[i * 2 + 1] = update[idx[i] * 2 + 1];
        }
        update = std::move(shuffled);
    }

    std::cout << "Graph loaded:\n"
              << "  vertices       : " << n_vertices << "\n"
              << "  directed edges : " << n_edges << "\n"
              << "  local_update edges : " << update.size() << "\n"
              << "  approx. undirected edges : " << (n_edges / 2) << "\n";
}

void CaLiG::constructCand(uint32_t num_batch)
{
    delta_ins_merged_.resize(num_batch);
    delta_del_merged_.resize(num_batch);
    delta_update_merged_.resize(num_batch);
    tbb::parallel_for(tbb::blocked_range<int>(0, G_size), [&](const tbb::blocked_range<int>& r) {
    for (int vi = r.begin(); vi != r.end(); vi++)
    {
        G_Li[vi] = 0;
        int &lb = G[vi].label;
        if (lb == -1)
            continue;
        for (auto &ui : labels[lb])
        {
            ska::flat_hash_map<int, u_set> ui_cand;
            for (auto &uj : Q[ui].nei) {
                ui_cand[uj] = u_set();
            }
            for (auto &vj : G[vi].nei)
            {
                int vj_lb = G[vj].label;
                for (auto &uj : Q[ui].nei)
                    if (uj >= 0 && vj >= 0 && Q[uj].label == vj_lb)
                        ui_cand[uj].insert(vj);
            }
            G[vi].cand[ui] = ui_cand;
            writeOr(&G_Li[vi], (1u << ui));
            G[vi].isChecked[ui] = 0;
        }
    }
    }, tbb::static_partitioner());
}

bool CaLiG::tryNei(int th, int vi, int ui, const CandMap& cand_ui, u_set &used, vec &to_check)
{
#ifdef COUNT_TRYNEI
    int tid = tbb::this_task_arena::current_thread_index();
    tryNei_count_[tid]++;
#endif
    if (th == to_check.size())
        return 1;
    auto &uj = to_check[th];
    auto it = cand_ui.find(uj);
    if (it == cand_ui.end())
        exit(-1);
    for (auto &vj : it->second)
    {
        if (G[vj].label != Q[uj].label)
            continue;
        if (used.find(vj) == used.end())
        {
            used.insert(vj);
            if (tryNei(th + 1, vi, ui, cand_ui, used, to_check))
                return 1;
            used.erase(vj);
        }
    }
    return 0;
}

bool CaLiG::checkNei(int vi, int ui)
{
#ifdef COUNT_TRYNEI
    int tid = tbb::this_task_arena::current_thread_index();
    tryNei_count_[tid]++;
#endif
    auto it_ui = G[vi].cand.find(ui);
    if (it_ui == G[vi].cand.end())
        exit(-1);
    auto& cand_ui = it_ui->second;
    for (auto &uj : Q[ui].nei)
    {
        auto it_uj = cand_ui.find(uj);
        if (it_uj == cand_ui.end() || it_uj->second.empty())
            return 0;
    }
    for (auto &rep_nei : Q[ui].rep_nei)
    {
        u_set used;
        if (!tryNei(0, vi, ui, cand_ui, used, rep_nei.second))
            return 0;
    }
    return 1;
}

void CaLiG::delAndCheck(vector<Task> &global, vector<vector<Task>> &local)
{
    tbb::parallel_for(0, (int)global.size(), [&](int i) {
        int tid = tbb::this_task_arena::current_thread_index(), ui = global[i].ui, vi = global[i].vi;
        for (auto &nei : G[vi].cand.at(ui))
        {
            int uj = nei.first;
            for (auto &vj : nei.second)
            {
                if (G[vj].label != Q[uj].label)
                    continue;
                if (G_Li[vj] & (1 << uj))
                    local_tmp_[tid].push_back({ui, uj, vi, vj});
            }
        }
    });

    mergeTasks<TmpTask>(local_tmp_, global_tmp_);

    tbb::parallel_for(0, (int)global_tmp_.size(), [&](int i) {
        int ui = global_tmp_[i].ui, uj = global_tmp_[i].uj;
        int vi = global_tmp_[i].vi, vj = global_tmp_[i].vj;
        int tid = tbb::this_task_arena::current_thread_index();
        lockVertex(vj, uj, ui);
        if(G[vj].cand[uj][ui].erase(vi))
            delta_del_local_[tid].push_back({(uint32_t)vj, (uint32_t)vi, (uint32_t)uj, (uint32_t)ui});
        bool is_empty = G[vj].cand[uj][ui].empty();
        unlockVertex(vj, uj, ui);
        if (is_empty)
        {
            writeAnd(&G_Li[vj], ~(1u << uj));
            li_changed_local_[tid].push_back(vj);
            local[tid].push_back({uj, vj});
        }
        else
        {
            auto lb = Q[ui].label;
            if (Q[uj].rep_nei.find(lb) != Q[uj].rep_nei.end()) {
                local_check_del_[tid].push_back({uj, vj, lb});
                if(G[vj].isChecked[uj])
                    writeAnd(&(G[vj].isChecked[uj]), false);
            }
        }
    });

    mergeTasks<DelTask>(local_check_del_, global_check_del_);

    tbb::parallel_for(0, (int)global_check_del_.size(), [&](int i) {
        int tid = tbb::this_task_arena::current_thread_index();
        int uj = global_check_del_[i].ui, vj = global_check_del_[i].vi, lb = global_check_del_[i].lb;
        if(G[vj].isChecked[uj] || writeOr(&(G[vj].isChecked[uj]), true))
            return;
        auto &rep_nei = Q[uj].rep_nei[lb];
        u_set used;
        auto it_uj = G[vj].cand.find(uj);
        if (it_uj == G[vj].cand.end())
            exit(-1);
        auto& cand_uj = it_uj->second;
        if (!tryNei(0, vj, uj, cand_uj, used, rep_nei))
        {
            writeAnd(&G_Li[vj], ~(1u << uj));
            li_changed_local_[tid].push_back(vj);
            local[tid].push_back({uj, vj});
        }
    });
}

void CaLiG::turnOffCond(int v1, int v2, vector<vector<int>> &local_check, vector<vector<TmpTask>> local_tmp)
{
    int tid = tbb::this_task_arena::current_thread_index();
    for (auto &candi : G[v1].cand)
    {
        int ui = candi.first;
        if (G[v1].label != Q[ui].label)
            continue;
        for (auto &ui_nei : candi.second)
        {
            int uj = ui_nei.first;
            if (Q[uj].label == G[v2].label)
            {
                local_tmp[tid].push_back({ui, uj, v1, v2});
                local_tmp[tid].push_back({uj, ui, v2, v1});
            }
        }
    }
    local_check[tid].push_back(v1);
    local_check[tid].push_back(v2);
}

void CaLiG::turnOff(int &vi, vector<vector<Task>> &local_del)
{
    int tid = tbb::this_task_arena::current_thread_index();
    for (auto &candi : G[vi].cand)
    {
        auto &ui = candi.first;
        if (G[vi].label != Q[ui].label)
            continue;
        if (!(G_Li[vi] & (1 << ui)))
            continue;
        if (!checkNei(vi, ui))
        {
            writeAnd(&G_Li[vi], ~(1u << ui));
            li_changed_local_[tid].push_back(vi);
            local_del[tid].push_back({ui, vi});
        }
    }
}

void CaLiG::turnOffCheck(vector<int> &global_check, vector<TmpTask> &global_tmp, vector<vector<Task>> &local_del)
{
    tbb::parallel_for(0, (int)global_tmp.size(), [&](int i) {
        int tid = tbb::this_task_arena::current_thread_index();
        int ui = global_tmp[i].ui, uj = global_tmp[i].uj;
        int vi = global_tmp[i].vi, vj = global_tmp[i].vj;

        lockVertex(vi, ui, uj);
        if(G[vi].cand[ui][uj].erase(vj))
            delta_del_local_[tid].push_back({(uint32_t)vi, (uint32_t)vj, (uint32_t)ui, (uint32_t)uj});
        unlockVertex(vi, ui, uj);
    });

    tbb::parallel_for(0, (int)global_check.size(), [&](int i) {
        int vi = global_check[i];
        turnOff(vi, local_del);
    });
}
void CaLiG::addAndCheck(vector<Task> &global, vector<vector<Task>> &local, vector<vector<Task>> &local_add_del)
{
    if(global.size() >= 100000) {
        tbb::parallel_for(tbb::blocked_range<int>(0, global.size(), 10000), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); i++)
        {
            int tid = tbb::this_task_arena::current_thread_index(), ui = global[i].ui, vi = global[i].vi;
            for (auto &nei : G[vi].cand.at(ui))
            {
                int uj = nei.first;
                for (auto &vj : nei.second)
                {
                    int lock_id = getLockId(vj, uj, ui);
                    if (G[vj].cand[uj][ui].find(vi) == G[vj].cand[uj][ui].end()) {
                        lockVertex(vj, uj, ui);
                        bucket_[lock_id].push_back({ui, uj, vi, vj});
                        unlockVertex(vj, uj, ui);
                    }
                    if (!(G_Li[vj] & (1 << uj))) {
                        local_check_[tid].push_back({uj, vj});
                        if(G[vj].isChecked[uj])
                            writeAnd(&G[vj].isChecked[uj], false);
                    }
                }
            }
        }
        }, tbb::auto_partitioner());

        tbb::parallel_for(tbb::blocked_range<int>(0, LOCK_COUNT), [&](const tbb::blocked_range<int>& r) {
        int tid = tbb::this_task_arena::current_thread_index();
        for (int lock_id = r.begin(); lock_id != r.end(); lock_id++)
        {
            for (auto& task : bucket_[lock_id])
            {
                if (G[task.vj].cand[task.uj][task.ui].find(task.vi) == G[task.vj].cand[task.uj][task.ui].end()) {
                    if(G[task.vj].cand[task.uj][task.ui].insert(task.vi).second)
                        delta_ins_local_[tid].push_back({(uint32_t)task.vj, (uint32_t)task.vi, (uint32_t)task.uj, (uint32_t)task.ui});
                }
            }
            bucket_[lock_id].clear();
        }
        }, tbb::static_partitioner());
    } else {
        tbb::parallel_for(tbb::blocked_range<int>(0, global.size(), 10000), [&](const tbb::blocked_range<int>& r) {
        for (int i = r.begin(); i != r.end(); i++)
        {
            int tid = tbb::this_task_arena::current_thread_index(), ui = global[i].ui, vi = global[i].vi;
            for (auto &nei : G[vi].cand.at(ui))
            {
                int uj = nei.first;
                for (auto &vj : nei.second)
                {
                    if (G[vj].label != Q[uj].label)
                        continue;
                    local_tmp_[tid].push_back({ui, uj, vi, vj});
                    if (!(G_Li[vj] & (1 << uj))) {
                        local_check_[tid].push_back({uj, vj});
                        if(G[vj].isChecked[uj])
                            writeAnd(&G[vj].isChecked[uj], false);
                    }
                }
            }
        }
        }, tbb::auto_partitioner());

        mergeTasks<TmpTask>(local_tmp_, global_tmp_);
        tbb::parallel_for(tbb::blocked_range<int>(0, global_tmp_.size(), 10000), [&](const tbb::blocked_range<int>& r) {
        int tid = tbb::this_task_arena::current_thread_index();
        for (int i = r.begin(); i != r.end(); i++)
        {
            int ui = global_tmp_[i].ui, uj = global_tmp_[i].uj;
            int vi = global_tmp_[i].vi, vj = global_tmp_[i].vj;
            lockVertex(vj, uj, ui);
            if(G[vj].cand[uj][ui].insert(vi).second)
                delta_ins_local_[tid].push_back({(uint32_t)vj, (uint32_t)vi, (uint32_t)uj, (uint32_t)ui});
            unlockVertex(vj, uj, ui);
        }
        }, tbb::auto_partitioner());
    }

    mergeTasks<Task>(local_check_, global_check_);

    tbb::parallel_for(tbb::blocked_range<int>(0, global_check_.size(), 10000), [&](const tbb::blocked_range<int>& r) {
    for (int i = r.begin(); i != r.end(); i++)
    {
        int vj = global_check_[i].vi, uj = global_check_[i].ui;
        int tid = tbb::this_task_arena::current_thread_index();
        if(G[vj].isChecked[uj] || writeOr(&(G[vj].isChecked[uj]), true))
            continue;
        if (checkNei(vj, uj))
        {
            writeOr(&G_Li[vj], (1u << uj));
            li_changed_local_[tid].push_back(vj);
            local[tid].push_back({uj, vj});
            li_activated_queue[tid].push_back({uj, vj});
        }
        else
        {
            local_add_del[tid].push_back({uj, vj});
        }
    }
    }, tbb::auto_partitioner());
}

void CaLiG::turnOnCond(int &v1, int &v2, vector<vector<Task>> &local, vector<vector<TmpTask>> &local_tmp)
{
    int tid = tbb::this_task_arena::current_thread_index();
    for (auto &candi : G[v1].cand)
    {
        int ui = candi.first;
        if (G[v1].label != Q[ui].label)
            continue;
        for (auto &ui_nei : candi.second)
        {
            int uj = ui_nei.first;
            if (G[v2].label == Q[uj].label)
                local_tmp[tid].push_back({ui, uj, v1, v2});
        }
    }
}

void CaLiG::turnOnCheck(vector<TmpTask> &global_tmp, vector<vector<Task>> &local_add)
{
    tbb::parallel_for(0, (int)global_tmp.size(), [&](int i) {
        int ui = global_tmp[i].ui, uj = global_tmp[i].uj;
        int v1 = global_tmp[i].vi, v2 = global_tmp[i].vj;
        int tid = tbb::this_task_arena::current_thread_index();
        lockVertex(v1, ui, uj);
        if(G[v1].cand[ui][uj].find(v2) == G[v1].cand[ui][uj].end()) {
            G[v1].cand[ui][uj].insert(v2);
        }
        unlockVertex(v1, ui, uj);
        if (G_Li[v1] & (1 << ui))
            local_add[tid].push_back({ui, v1});
        else {
            local_check_[tid].push_back({ui, v1});
            if(G[v1].isChecked[ui])
                writeAnd(&(G[v1].isChecked[ui]), false);
        }
    });

    mergeTasks<Task>(local_check_, global_check_);
    tbb::parallel_for(0, (int)global_check_.size(), [&](int i) {
        int tid = tbb::this_task_arena::current_thread_index();
        int v1 = global_check_[i].vi, ui = global_check_[i].ui;
        if(G[v1].isChecked[ui] || writeOr(&(G[v1].isChecked[ui]), true))
            return;
        if(G_Li[v1] & (1 << ui))
            return;
        if (checkNei(v1, ui))
        {
            writeOr(&G_Li[v1], (1u << ui));
            li_changed_local_[tid].push_back(v1);
            local_add[tid].push_back({ui, v1});
            li_activated_queue[tid].push_back({ui, v1});
        }
    });
}

void CaLiG::addAndCheckDelete(vector<Task> &global_check_del, vector<vector<Task>> &local_del)
{
    tbb::parallel_for(0, (int)global_check_del.size(), [&](int i) {
        int tid = tbb::this_task_arena::current_thread_index();
        int v = global_check_del[i].vi, u = global_check_del[i].ui;
        if (!(G_Li[v] & (1 << u)))
            local_del[tid].push_back({u, v});
    });
}

void CaLiG::staticFilter()
{
    tbb::parallel_for(0, G_size, [&](int vi) {
        turnOff(vi, del_queue);
    });

    mergeTasks<Task>(del_queue, global_del_queue);
    while (!global_del_queue.empty())
    {
        delAndCheck(global_del_queue, del_queue);
        mergeTasks<Task>(del_queue, global_del_queue);
    }
}

void CaLiG::inputUpdate(string &path, int max_num)
{
    update.clear();
    ifstream infile(path);
    char c;
    int v1, v2, w;
    int cnt = 0;
    while (infile >> c >> v1 >> v2 >> w)
    {
        if (max_num != 0 && ++cnt > max_num)
            break;
        update.push_back(v1);
        update.push_back(v2);
    }
}

void CaLiG::count_candidate()
{
    vector<int> h_compacted_vs_sizes(Q_size, 0);
    for (int i = 0; i < G_size; i++)
        for (auto const &[u_id, neighbors_map] : G[i].cand)
            if (G_Li[i] & (1 << u_id))
                h_compacted_vs_sizes[u_id]++;

    map<pair<int, int>, long long> cardinalities;
    for (int v = 0; v < G_size; v++) {
        if (G_Li[v] == 0) continue;
        for (auto const& [u, neighbors_map] : G[v].cand) {
            if (!(G_Li[v] & (1 << u))) continue;
            for (auto const& [u_other, v_neighbors] : neighbors_map) {
                for (int v_neighbor : v_neighbors) {
                    if (G_Li[v_neighbor] & (1 << u_other)) {
                        cardinalities[{u, u_other}]++;
                    }
                }
            }
        }
    }

    cout << "\n# Candidate edges (CaLiG Style): \n";
    for (int u = 0; u < Q_size; u++)
    {
        for (int u_other : Q[u].nei)
        {
            if (u > u_other)
                continue;
            pair<int, int> edge = {u, u_other};
            pair<int, int> rev_edge = {u_other, u};
            cout << "(" << u << ", " << u_other << "): "
                 << cardinalities[edge] << " "
                 << u << ": " << h_compacted_vs_sizes[u] << " "
                 << u_other << ": " << h_compacted_vs_sizes[u_other] << "\n";
        }
    }
    cout << endl;
}

void CaLiG::ConstructUpdate(uint32_t start, uint32_t count)
{
    uint32_t end = std::min(start + count, static_cast<uint32_t>(update.size()));
    int num_pairs = (end - start) / 2;
    tbb::parallel_for(0, num_pairs, [&](int idx) {
        int t = start + idx * 2;
        int tid = tbb::this_task_arena::current_thread_index(), v1 = update[t], v2 = update[t + 1];
        for (auto &[u1, candij] : G[v1].cand)
        {
            if (!(G_Li[v1] & (1u << u1)))
                continue;
            for (auto &candi : candij)
            {
                int u2 = candi.first;
                if (u2 >= 0 && candi.second.find(v2) != candi.second.end())
                {
                    lockVertex(v1, u1, u2);
                    delta_update_local_[tid].push_back({(uint32_t)v1, (uint32_t)v2, (uint32_t)u1, (uint32_t)u2});
                    delta_del_local_[tid].push_back({(uint32_t)v1, (uint32_t)v2, (uint32_t)u1, (uint32_t)u2});
                    unlockVertex(v1, u1, u2);
                    lockVertex(v2, u2, u1);
                    delta_update_local_[tid].push_back({(uint32_t)v2, (uint32_t)v1, (uint32_t)u2, (uint32_t)u1});
                    delta_del_local_[tid].push_back({(uint32_t)v2, (uint32_t)v1, (uint32_t)u2, (uint32_t)u1});
                    unlockVertex(v2, u2, u1);
                }
            }
        }
    });
}

// 1-hop, per-pattern-edge conflict detection.
//
// For each pattern edge ei (undirected, numbered as in QueryGraph::qe_list_ so it
// matches the GPU's g_d_cf / C_REBUILD_V_FLAGS exactly), independently look at the
// batch of update edges realizing ei and detect collisions on their 1-hop candidate
// slots (qv, dv): two different update edges claiming the same owner3d[ei][qv][dv]
// slot => conflict.
//
// Output: edge_conflict_[ei * num_pairs + k] = 1 if update edge k (realizing ei) is
// involved in such a collision. CaLiGHelper::BuildCompressedConflictFree2D later
// maps each conflicting edge's two endpoints (v1_k, v2_k) to the per-vertex GPU
// structure (g_d_cf). The large DV_COUNT*QE_COUNT per-vertex array is avoided.
//
// tag = bbase + k (k = update-edge index within this batch): no atomic tag
// generation. dc_owner_ (owner3d[ei][qv][dv]) is reused across batches via the
// generation tag bbase (slots holding a value < bbase are stale), so the huge owner
// array is never cleared per batch. Only claim_owner uses CAS; edge_conflict_ writes
// are atomic relaxed stores of 1. Different threads can target the SAME slot — thread
// k writes [base+k] itself, while any other thread that finds k owns a slot writes
// [base+k] via conflict_with(ok) — so the store must be atomic to be a well-defined
// (race-free) concurrent same-value write under the C++ memory model.
// void CaLiG::DetectConflict(uint32_t start, uint32_t count)
// {
//     dc_reached_ = dc_conflicted_ = 0;

//     const int qsize = static_cast<int>(Q.size());
//     const int gsize = static_cast<int>(G.size());
//     if (qsize == 0 || gsize == 0 || dc_qg_ == nullptr) return;

//     // Build (u0,u1) -> undirected ei once, reading the GPU's qe_list_ directly so
//     // the ei numbering matches g_d_cf / C_REBUILD_V_FLAGS (verified: those are
//     // indexed by the undirected qe_list_ position, not by the directed eidx_).
//     if (dc_ei_of_.empty()) {
//         dc_ei_of_.assign(static_cast<size_t>(qsize) * qsize, UINT32_MAX);
//         for (uint32_t e = 0; e < dc_qg_->ecount_; ++e) {
//             uint32_t u = dc_qg_->qe_list_[e].first;
//             uint32_t uu = dc_qg_->qe_list_[e].second;
//             dc_ei_of_[static_cast<size_t>(u) * qsize + uu] = e;
//             dc_ei_of_[static_cast<size_t>(uu) * qsize + u] = e;  // undirected: both dirs -> same ei
//         }
//         dc_ecount_ = dc_qg_->ecount_;
//     }
//     const uint32_t ecount = dc_ecount_;

//     const uint32_t end = std::min(start + count, static_cast<uint32_t>(update.size()));
//     const int num_pairs = static_cast<int>((end - start) / 2);
//     dc_start_ = start;
//     dc_num_pairs_ = static_cast<uint32_t>(num_pairs);
//     if (num_pairs <= 0) { edge_conflict_.clear(); return; }

//     // Output: [ei][k], default 0 (no conflict).
//     edge_conflict_.assign(static_cast<size_t>(ecount) * num_pairs, 0);

//     // owner3d[ei][qv][dv]: reused across batches via generation tag (no clear).
//     const size_t owner_total = static_cast<size_t>(ecount) * qsize * gsize;
//     if (dc_owner_.size() != owner_total) {
//         dc_owner_.assign(owner_total, 0u);
//         dc_tag_counter_ = 1ull;
//     }
//     // Wrap guard: keep every tag of this batch safely below the 0xFF000000 ceiling.
//     if (dc_tag_counter_ + static_cast<uint64_t>(num_pairs) >= 0xFF000000ull) {
//         std::fill(dc_owner_.begin(), dc_owner_.end(), 0u);
//         dc_tag_counter_ = 1ull;
//     }
//     const uint32_t bbase = static_cast<uint32_t>(dc_tag_counter_);

//     const size_t QS = static_cast<size_t>(qsize);
//     const size_t GS = static_cast<size_t>(gsize);
//     auto slot3d = [&](uint32_t ei, int qv, int dv) -> size_t {
//         return static_cast<size_t>(ei) * QS * GS + static_cast<size_t>(qv) * GS + dv;
//     };
//     // 0 = newly claimed, 1 = same tag (same update edge), 2 = different tag (conflict)
//     auto claim_owner = [&](size_t p, uint32_t t) -> int {
//         uint32_t old = dc_owner_[p];
//         while (old < bbase) {
//             if (__sync_bool_compare_and_swap(&dc_owner_[p], old, t)) return 0;
//             old = dc_owner_[p];
//         }
//         return (old == t) ? 1 : 2;
//     };

//     std::atomic<uint64_t> a_reached(0), a_conf(0);

//     tbb::parallel_for(0, num_pairs, [&](int k) {
//         const int t = start + k * 2;
//         const int v1 = update[t];
//         const int v2 = update[t + 1];
//         const uint32_t tag = bbase + static_cast<uint32_t>(k);

//         for (auto const& kv_u0 : G[v1].cand) {
//             const int u0 = kv_u0.first;
//             if (!(G_Li[v1] & (1u << u0))) continue;
//             auto const& cand_u0 = kv_u0.second;
//             for (auto const& kv_u1 : cand_u0) {
//                 const int u1 = kv_u1.first;
//                 if (!(G_Li[v2] & (1u << u1))) continue;
//                 auto const& dvset = kv_u1.second;
//                 if (dvset.find(v2) == dvset.end()) continue;
//                 const uint32_t ei = dc_ei_of_[static_cast<size_t>(u0) * qsize + u1];

//                 const size_t base = static_cast<size_t>(ei) * num_pairs;
//                 auto conflict_with = [&](size_t p) {
//                     __atomic_store_n(&edge_conflict_[base + k], 1, __ATOMIC_RELAXED);
//                     const uint32_t other = dc_owner_[p];
//                     if (other >= bbase) {
//                         const uint32_t ok = other - bbase;
//                         if (ok < static_cast<uint32_t>(num_pairs))
//                             __atomic_store_n(&edge_conflict_[base + ok], 1, __ATOMIC_RELAXED);
//                     }
//                 };

//                 for (auto const& kv_qv : cand_u0) {
//                     const int qv = kv_qv.first;
//                     for (const int dv : kv_qv.second) {
//                         if(ei == 7 && (dv == 43208 || dv == 1153503))
//                             printf("ei %d qv %d dv %d v1 %d u0 %d\n", ei, qv, dv, v1, u0);
//                         const int r = claim_owner(slot3d(ei, qv, dv), tag);
//                         if (r == 0) a_reached.fetch_add(1, std::memory_order_relaxed);
//                         else if (r == 2) { conflict_with(slot3d(ei, qv, dv)); a_conf.fetch_add(1, std::memory_order_relaxed); }
//                     }
//                 }

//                 auto itu = G[v2].cand.find(u1);
//                 if (itu != G[v2].cand.end()) {
//                     for (auto const& kv_qv : itu->second) {
//                         const int qv = kv_qv.first;
//                         for (const int dv : kv_qv.second) {
//                             if(ei == 7 && (dv == 43208 || dv == 1153503))
//                                 printf("ei %d qv %d dv %d v2 %d u1 %d\n", ei, qv, dv, v2, u1);
//                             const int r = claim_owner(slot3d(ei, qv, dv), tag);
//                             if (r == 0) a_reached.fetch_add(1, std::memory_order_relaxed);
//                             else if (r == 2) { conflict_with(slot3d(ei, qv, dv)); a_conf.fetch_add(1, std::memory_order_relaxed); }
//                         }
//                     }
//                 }
//             }
//         }
//     });

//     dc_tag_counter_ = static_cast<uint64_t>(bbase) + static_cast<uint64_t>(num_pairs) + 1ull;
//     dc_reached_ = a_reached.load();
//     dc_conflicted_ = a_conf.load();
// }

void CaLiG::updateIndex(uint32_t batch_start, uint32_t batch_count, uint32_t batch)
{
    this->batch = batch;
    for (auto& v : li_changed_local_) v.clear();
    for (auto& v : delta_ins_local_) v.clear();
    for (auto& v : delta_del_local_) v.clear();
    for (auto& v : delta_update_local_) v.clear();
    delta_ins_merged_.clear();
    delta_del_merged_.clear();
    delta_update_merged_.clear();
    li_changed_global_.clear();
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff;

    // auto tick = [&]() { start_time = std::chrono::high_resolution_clock::now(); };
    // auto tock = [&](const char* name) {
    //     end_time = std::chrono::high_resolution_clock::now();
    //     diff = end_time - start_time;
    //     std::cout << name << ", time (ms): " << (unsigned long)diff.count() << "(host)\n";
    // };

    // tick();
    uint32_t update_end = std::min(batch_start + batch_count, static_cast<uint32_t>(update.size()));
    int num_pairs = (update_end - batch_start) / 2;
    tbb::parallel_for(0, num_pairs, [&](int idx) {
        int t = batch_start + idx * 2;
        int tid = tbb::this_task_arena::current_thread_index();
        int v1 = update[t];
        int v2 = update[t + 1];
        if (v1 < 0)
        {
            int v11 = -v1 - 1, v22 = -v2 - 1;
            if (G[v11].label == -1 || G[v22].label == -1) return;
            if (!G[v11].nei.count(v22)) return;
            update_queue[tid].push_back({v11, v22, -1});
            update_queue[tid].push_back({v22, v11, -1});
            turnOffCond(v11, v22, del_check_queue, tmp_queue_1);
            turnOffCond(v22, v11, del_check_queue, tmp_queue_1);
        }
        else
        {
            if (G[v1].label == -1 || G[v2].label == -1) return;
            if (G[v1].nei.count(v2)) return;
            update_queue[tid].push_back({v1, v2, 1});
            update_queue[tid].push_back({v2, v1, 1});
            turnOnCond(v1, v2, add_queue, tmp_queue);
            turnOnCond(v2, v1, add_queue, tmp_queue);
        }
    });
    // tock("Step 2: Process update queue");

    // tick();
    mergeTasks<DelTask>(update_queue, global_update_queue);
    // tock("Step 3: Merge DelTask");

    // tick();
    tbb::parallel_for(0, (int)global_update_queue.size(), [&](int i) {
        int v1 = global_update_queue[i].ui, v2 = global_update_queue[i].vi, lb = global_update_queue[i].lb;
        lockVertex(v1);
        if (lb == 1) G[v1].nei.insert(v2);
        else G[v1].nei.erase(v2);
        unlockVertex(v1);
    });
    // tock("Step 4: Update G.nei");

    // tick();
    mergeTasks<TmpTask>(tmp_queue, global_tmp_queue);
    turnOnCheck(global_tmp_queue, add_queue);
    // tock("Step 5: turnOnCheck");

    // tick();
    mergeTasks<TmpTask>(tmp_queue_1, global_tmp_queue_1);
    mergeTasks<int>(del_check_queue, global_del_check_queue);
    turnOffCheck(global_del_check_queue, global_tmp_queue_1, del_queue);
    // tock("Step 6: turnOffCheck");

    // tick();
    mergeTasks<Task>(add_queue, global_add_queue);
    while (!global_add_queue.empty())
    {
        addAndCheck(global_add_queue, add_queue, add_del_check_queue);
        mergeTasks<Task>(add_queue, global_add_queue);
    }
    mergeTasks<Task>(add_del_check_queue, global_add_del_check_queue);
    addAndCheckDelete(global_add_del_check_queue, del_queue);
    // tock("Step 7: addAndCheck");

    // tick();
    mergeTasks<Task>(del_queue, global_del_queue);
    while (!global_del_queue.empty())
    {
        delAndCheck(global_del_queue, del_queue);
        mergeTasks<Task>(del_queue, global_del_queue);
    }
    // tock("Step 8: delAndCheck");

    // tick();
    mergeTasks<Task>(li_activated_queue, global_li_activated);
    mergeTasks<uint32_t>(li_changed_local_, li_changed_global_);
    // tock("Step 9: Merge li_activated");

    // tick();
    tbb::parallel_for(0, (int)global_li_activated.size(), [&](int i) {
        int tid = tbb::this_task_arena::current_thread_index();
        int vi = global_li_activated[i].vi;
        int ui = global_li_activated[i].ui;
        if (!(G_Li[vi] & (1 << ui))) return;
        if (G[vi].cand.find(ui) == G[vi].cand.end()) return;
        for (auto &cand_pair : G[vi].cand[ui])
        {
            int uj = cand_pair.first;
            for (int vj : cand_pair.second)
            {
                delta_ins_local_[tid].push_back({(uint32_t)vi, (uint32_t)vj, (uint32_t)ui, (uint32_t)uj});
            }
        }
    });
    // tock("Step 10: Record delta_ins");
}

void CaLiG::mergeDeltas(uint32_t batch)
{
    auto& ins = delta_ins_merged_[batch];
    auto& del = delta_del_merged_[batch];
    auto& upd = delta_update_merged_[batch];
    ins.clear();
    del.clear();
    upd.clear();
    for (auto& l : delta_ins_local_)
        ins.insert(ins.end(), l.begin(), l.end());
    for (auto& l : delta_del_local_)
        del.insert(del.end(), l.begin(), l.end());
    for (auto& l : delta_update_local_)
        upd.insert(upd.end(), l.begin(), l.end());
    for (auto& v : delta_ins_local_) v.clear();
    for (auto& v : delta_del_local_) v.clear();
    for (auto& v : delta_update_local_) v.clear();
}

CaLiG::CaLiG()
{
    int num_numa = getNumNumaNodes();
    int total = std::thread::hardware_concurrency();
    num_threads_ = total / std::max(num_numa, 1);
    if (num_threads_ <= 0) num_threads_ = total;
    int T = num_threads_;

    li_activated_queue.resize(T);
    li_changed_local_.resize(T);
    update_queue.resize(T);
    add_queue.resize(T);
    del_queue.resize(T);
    check_queue.resize(T);
    add_del_check_queue.resize(T);
    del_check_queue.resize(T);
    tmp_queue.resize(T);
    tmp_queue_1.resize(T);
    local_tmp_.resize(T);
    local_check_.resize(T);
    local_check_del_.resize(T);
    for(int i = 0; i < T; i++) {
        local_tmp_[i].reserve(10000000);
        local_check_[i].reserve(10000000);
        local_check_del_[i].reserve(10000000);
    }
    global_tmp_.reserve(10000000 * T);
    global_check_.reserve(10000000 * T);
    global_check_del_.reserve(10000000 * T);
    bucket_.resize(LOCK_COUNT);
    for (int i = 0; i < LOCK_COUNT; i++)
        bucket_[i].reserve(1000);

    delta_ins_local_.resize(T);
    delta_del_local_.resize(T);
    delta_update_local_.resize(T);
#ifdef COUNT_TRYNEI
    tryNei_count_.resize(T, 0);
#endif
}

CaLiG::~CaLiG()
{
}
