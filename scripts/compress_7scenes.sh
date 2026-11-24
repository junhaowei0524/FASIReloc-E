#!/bin/bash
set -e

for scene in chess fire heads office pumpkin redkitchen stairs
do
    echo "Compressing ${scene}"

    python ./sapc/vectree_split.py \
        --mode compress \
        --input_path map/7scenes/${scene}/point_cloud/iteration_30000/point_cloud.ply \
        --important_score_npz_path map/7scenes/${scene}/detector \
        --save_path map/7scenes/${scene}/compressed \
        --codebook_size 8192 \
        --iteration_num 1000
done