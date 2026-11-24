#!/bin/bash
set -e

for scene in chess fire heads office pumpkin redkitchen stairs
do
    echo "Decompressing ${scene}"

    python ./vectree_split.py \
        --mode decompress \
        --save_path map/7scenes/${scene}/compressed
done