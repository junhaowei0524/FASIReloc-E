#!/bin/bash
set -e

for scene in GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch; do
    echo "========== Training ${scene} =========="

    python train.py \
        -s "datasets/cambridge/${scene}" \
        -m "map/cambridge/${scene}" \
        -r 1 \
        -f sp \
        -g fgs \
        --iterations 30000 \
        --data_device cpu \
        --train_detector \
        --densify_grad_threshold 0.0004 \
        --images "processed" \
        --position_lr_init 0.000016 \
        --scaling_lr 0.001 \
        --feat_dim 32
done

echo "All Cambridge Landmarks training completed."