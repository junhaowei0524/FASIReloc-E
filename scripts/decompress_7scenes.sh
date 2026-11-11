#!/bin/bash
set -e

for scene in chess fire heads office pumpkin redkitchen stairs; do
    python decompress_map.py \
        --input_path "compressed/7scenes/${scene}/extreme_saving.zip" \
        --save_path "map/7scenes/${scene}/point_cloud/iteration_30000/"
done