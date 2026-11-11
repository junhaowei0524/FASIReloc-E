#!/bin/bash
set -e

export PYTHONPATH="JamMa:$PYTHONPATH"

for scene in chess fire heads office pumpkin redkitchen stairs; do
    echo "========== Evaluating ${scene} =========="

    python fasireloc_e_dependent.py \
        -m "map/7scenes/${scene}" \
        --cfg configs/fasireloc_e_indoor.yaml \
        --main_cfg_path JamMa/configs/jamma/indoor/test.py
done

echo "All 7 Scenes evaluations completed."