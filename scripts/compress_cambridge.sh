#!/bin/bash
set -e

for scene in GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch
do
    echo "Compressing Cambridge/${scene}"

    python ./sapc/vectree_split.py \
        --mode compress \
        --input_path map/cambridge/${scene}/point_cloud/iteration_30000/point_cloud.ply \
        --important_score_npz_path map/cambridge/${scene}/detector \
        --save_path map/cambridge/${scene}/compressed \
        --codebook_size 8192 \
        --iteration_num 1000
done