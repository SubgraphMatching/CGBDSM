# CGBDSM: CPU–GPU Co-processing Batched Dynamic Subgraph Matching

CGBDSM answers **continuous subgraph matching** (CSM) over edge-insertion
streams with a two-stage CPU–GPU pipeline:

- **Stage 1 (CPU, NUMA-pinned TBB arena)** maintains the CaLiG auxiliary index
  (`updateIndex` → `ConstructUpdate` → `mergeDeltas`) and prepares per-batch
  update structures.
- **Stage 2 (GPU)** transfers the batch, rebuilds the affected index entries,
  and runs the merged matching kernel (`USE_MERGED_MATCHING`: all query edges
  share one kernel launch with a 5-bit edge tag; final matches are counted by
  an on-GPU DFS).

The two stages run as a pipelined producer/consumer over fixed-size update
batches; the program prints per-stage timings plus the total number of
positive matches.

# 1. Compiling

## Third-party Requirements

+ CUDA toolkit 12.x (developed against 12.9) with `nvcc`; GPU compute
  capability **7.5** by default (see notes below).

+ CMake >= 3.10, GCC with C++17 support.

+ Intel oneTBB (`tbb`, `tbbmalloc`, `tbbmalloc_proxy`) — the project expects
  oneTBB with NUMA support (2022.x).

+ libnuma (`numactl` package); linked into every target and used at runtime
  via `numactl --cpunodebind/--membind`.

+ Optional: [mimalloc](https://github.com/microsoft/mimalloc) for faster
  host-side allocation via `LD_PRELOAD` (not required to build or run).

## Adjust hardcoded paths in `CMakeLists.txt`

Three settings are hardcoded for the development machine and usually need
editing before the first build:

| Setting | Default | Change to |
|---|---|---|
| `include_directories(/usr/local/cuda-12.9/include)` (line 14) | CUDA 12.9 | your CUDA install path |
| `set(TBB_ROOT "/opt/intel/oneapi/tbb/2022.3")` (line 87) | oneTBB 2022.3 | your oneTBB install path |
| `CUDA_ARCHITECTURES "75"` (every target) | Turing (e.g. RTX 2080 Ti / TITAN RTX) | your GPU's compute capability (e.g. `80` for A100, `90` for H100) |

At link time every `cgcsm*` target needs `-lnuma`, `-lcudart`, and the three
TBB libraries; at runtime oneTBB must be on `LD_LIBRARY_PATH` (see §2.1).

## Build

```
mkdir build && cd build && cmake .. && make -j
```

## Build targets

All `cgcsm*` executables are built from the same sources (`main_calig.cpp` +
`index/` + `graph/`) with different compile definitions; they are variants /
ablations of the same pipeline:

| Target | Compile definitions | Description |
|---|---|---|
| `cgcsm` | — | Baseline: per-edge local-index rebuild + per-edge matching |
| `cgcsm_without_localindex` | `SKIP_BUILD_LOCAL_INDEX` | Ablation: skips local index build |
| `cgcsm_valid_bits` | `USE_VALID_BITS` | Sparse per-vertex valid bits instead of dense local index |
| `cgcsm_merged_matching` | `USE_VALID_BITS`, `USE_MERGED_MATCHING` | **Main variant**: one merged kernel launch for all edges (5-bit edge tag) |
| `cgcsm_merged_matching_global` | + `USE_GLOBAL_RQ` | Merged matching with a global result queue |
| `cgcsm_gamma` | `USE_GAMMA`, `USE_VALID_BITS`, `USE_MERGED_MATCHING` | GAMMA-style per-vertex sorted-merge batched index update |
| `cgcsm_cpu_dfs` | `USE_GAMMA` + `ENABLE_CPU_DFS` | Additionally mirrors the index to a second NUMA node and shares the final DFS workload with the CPU |
| `cgcsm_gpma` | `USE_GPMA`, `USE_VALID_BITS`, `USE_MERGED_MATCHING` | Index maintenance on GPMA (dynamic graph layout on GPU) |
| `cgcsm_default_order` | `FORCE_UPDATE_EDGE_START` | Forces matching order to start from the update edge |

Auxiliary tools (no GPU needed):

| Target | Description |
|---|---|
| `graph_sampler` | Generates tree / dense / sparse query sets from a data graph |
| `motivation` | Standalone CaLiG index-maintenance experiment with `tryNei` counting |

# 2. Running

## 2.1 Basic usage

> Usage: `./cgcsm_merged_matching <query_path> <data_path> <update_path> <gpu_id> [batch_size]`

+ `<query_path>`: pattern (query graph) file.

+ `<data_path>`: initial data graph, either a `.graph` text file (then
  `<update_path>` is read as the edge stream) or a binary CSR prefix (see
  §3.3; the update stream is then derived internally and `<update_path>` is
  ignored).

+ `<gpu_id>`: CUDA device index (e.g. `0`).

+ `[batch_size]`: updates per batch (default: all updates in a single batch,
  i.e. `UINT32_MAX`).

Recommended launch (bind CPU/memory to NUMA nodes, expose oneTBB libs,
optional mimalloc):

```
export LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2022.3/lib:$LD_LIBRARY_PATH
numactl --cpunodebind=0,1 --membind=0,1 \
    ./build/cgcsm_merged_matching \
    <query_path> <data_path> <update_path> 0 1000000
```

Example:

```
export LD_LIBRARY_PATH=/opt/intel/oneapi/tbb/2022.3/lib:$LD_LIBRARY_PATH && \
LD_PRELOAD=~/mimalloc/out/release/libmimalloc.so \
numactl --cpunodebind=0,1 --membind=0,1 \
    ./build/cgcsm_merged_matching \
    ~/data/livejournal/random_walk/7_self/dense/Q_9 \
    ~/data/livejournal/data_graph/data.graph \
    ~/data/livejournal/data_graph/insertion.graph 1 1000000
```

Environment variables:

+ `CGCSM_CPU_DFS_NUMA` (only `cgcsm_cpu_dfs`): NUMA node for the CPU-DFS
  arena and index mirror (default `2`; the Stage-1 node is avoided so CPU DFS
  gets dedicated cores).

At the end the program prints `Num Positive Matches` and a per-stage
profiling summary (`[S1] ...` CPU index maintenance, `[S2] ...` GPU transfer /
index update / matching, sync waits, and `Bottleneck = max(S1, S2)`).

## 2.2 Generating query sets

> Usage: `./graph_sampler <data_path> <output_dir> <sample_size> <num_samples> <type> [seed]`
> Supported types: `dense` | `sparse` | `tree` | `all`

Writes queries to `<output_dir>/<sample_size>_self/<type>/Q_<i>`. Example:

```
./graph_sampler ~/data/orkut/graph ~/data/orkut/query_graph 7 50 all
```

`sampler/run_tree.sh` shows a batch script over multiple datasets/sizes.

## 2.3 Index-maintenance motivation experiment

> Usage: `./motivation <query_path> <data_path> <update_path>`

Loads the graph, builds the CaLiG index, applies all updates once, and
reports `tryNei` invocation counts (compiled with `COUNT_TRYNEI`).

# 3. Data Format

## 3.1 Query / data graph (text, `.graph`)

```
v <id> <label>
v <id> <label>
...
e <id1> <id2> <label>
e <id1> <id2> <label>
...
```

Vertices first, then edges. Vertex ids in the data graph must be dense and
ascending (gaps are filled with invalid label `-1`); vertices whose label
does not occur in the query are pruned.

## 3.2 Update stream (text)

One update per line, 4 columns:

```
<char> <v1> <v2> <w>
```

Only the two vertex ids are used (edges are treated as insertions); the
leading char and trailing weight columns are ignored.

## 3.3 Binary CSR (optional fast path)

If `<data_path>` does not end with `.graph`, it is read as a binary CSR
prefix:

+ `<prefix>.meta.txt`: `n_vertices n_edges vsize esize vlbl_sz elbl_sz
  max_deg feat n_vlabel_classes n_elabel_classes` (requires `vsize=4`,
  `esize=8`)
+ `<prefix>.vertex.bin`: `uint64 row_ptr[n_vertices + 1]`
+ `<prefix>.edge.bin`: `uint32 edges[n_edges]`

Vertex labels are assigned pseudo-randomly over the query's label classes and
~10% of edges are deterministically split off (hash-based) as the update
stream. Intended for quick experiments without a prepared update file.

# 4. Repository Layout

```
main_calig.cpp        entry point of all cgcsm* executables
index/                CaLiG CPU index + GPU update helpers (gamma / gpma variants,
                      CPU-DFS mirror for cgcsm_cpu_dfs)
graph/                graph loaders, indexing/matching-order planner, GPU relations
kernels/              CUDA kernels: global/local index build, enumeration,
                      merged matching, cartesian product
utils/                compile-time config, memory pool, CUDA/NUMA helpers,
                      nucleus decomposition (utils/nucleus)
sampler/              standalone query-set generator (graph_sampler)
gpma/                 GPMA library (VLDB 2017), used by cgcsm_gpma
```

Compile-time tuning knobs live in `utils/config.h` (e.g. `SUPPORT_MASK_WIDTH`,
`ENABLE_WARP_STEALING`, `USE_GSI_ORDER`; see comments there).

# 5. Acknowledgements

+ The CPU index follows the CaLiG / Symbi line of continuous subgraph matching
  systems.
+ `gpma/` is adapted from [GPMA](http://www.vldb.org/pvldb/vol11/p107-sha.pdf)
  (VLDB 2017), MIT license.
