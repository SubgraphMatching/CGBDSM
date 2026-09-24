#!/bin/bash
# 重新生成各数据集的 Tree 查询图（DFS + 深度验证）
# 用法: bash run_tree.sh

cd /home/wangchunxiang/GCSM/CGCSM/build

SAMPLER=./graph_sampler
N_SAMPLES=50

# Friendster
for size in 6 7 8 9 10; do
    echo "===== Friendster ${size}_self ====="
    $SAMPLER /home/wangchunxiang/GCSM/inputs/friendster/data_graph/graph \
        /home/wangchunxiang/GCSM/inputs/friendster/query_graph \
        $size $N_SAMPLES tree
    echo
done

# Graph500 scale 24
for size in 6 7 8 9 10; do
    echo "===== Graph500_24 ${size}_self ====="
    $SAMPLER /home/wangchunxiang/GCSM/inputs/graph500_24/data_graph/graph \
        /home/wangchunxiang/GCSM/inputs/graph500_24/query_graph \
        $size $N_SAMPLES tree
    echo
done

# Graph500 scale 25
for size in 6 7 8 9 10; do
    echo "===== Graph500_25 ${size}_self ====="
    $SAMPLER /home/wangchunxiang/GCSM/inputs/graph500_25/data_graph/graph \
        /home/wangchunxiang/GCSM/inputs/graph500_25/query_graph \
        $size $N_SAMPLES tree
    echo
done

# Graph500 scale 26
for size in 6 7 8 9 10; do
    echo "===== Graph500_26 ${size}_self ====="
    $SAMPLER /home/wangchunxiang/GCSM/inputs/graph500_26/data_graph/graph \
        /home/wangchunxiang/GCSM/inputs/graph500_26/query_graph \
        $size $N_SAMPLES tree
    echo
done

# Graph500 scale 27
for size in 6 7 8 9 10; do
    echo "===== Graph500_27 ${size}_self ====="
    $SAMPLER /home/wangchunxiang/GCSM/inputs/graph500_27/data_graph/graph \
        /home/wangchunxiang/GCSM/inputs/graph500_27/query_graph \
        $size $N_SAMPLES tree
    echo
done

echo "===== All done! ====="
