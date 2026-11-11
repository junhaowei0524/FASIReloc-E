#!/bin/bash
set -e

export PYTHONPATH="JamMa:$PYTHONPATH"

for scene in GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch; do
    echo "========== Evaluating ${scene} =========="

    python fasireloc_e_dependent.py \
        -m "map/cambridge/${scene}" \
        --cfg configs/fasireloc_e_outdoor.yaml \
        --main_cfg_path JamMa/configs/jamma/outdoor/test.py
done

echo "All Cambridge Landmarks evaluations completed."