#!/bin/bash
set -e

for scene in chess fire heads office pumpkin redkitchen stairs; do
    echo "========== Training ${scene} =========="

    python train.py \
        -s "datasets/7scenes/${scene}" \
        -m "map/7scenes/${scene}" \
        --iterations 30000 \
        --data_device cpu \
        --train_detector \
        -f sp \
        -g fgs \
        --images "" \
        --feat_dim 32
done

echo "All 7 Scenes training completed."