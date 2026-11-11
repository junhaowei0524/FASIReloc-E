#!/bin/bash
set -e

for scene in KingsCollege OldHospital ShopFacade StMarysChurch GreatCourt; do
    python decompress_map.py \
        --input_path "compressed/cambridge/${scene}/extreme_saving.zip" \
        --save_path "map/cambridge_32/${scene}/point_cloud/iteration_30000/"
done