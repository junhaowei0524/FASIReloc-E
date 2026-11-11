export PYTHONPATH="JamMa:$PYTHONPATH"

python fasireloc_e_independent.py -m map/cambridge/GreatCourt --cfg configs/fasireloc_e_outdoor.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py --path map/megadepth
python fasireloc_e_independent.py -m map/cambridge/KingsCollege --cfg configs/fasireloc_e_outdoor.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py --path map/megadepth
python fasireloc_e_independent.py -m map/cambridge/OldHospital --cfg configs/fasireloc_e_outdoor.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py --path map/megadepth
python fasireloc_e_independent.py -m map/cambridge/ShopFacade --cfg configs/fasireloc_e_outdoor.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py --path map/megadepth
python fasireloc_e_independent.py -m map/cambridge/StMarysChurch --cfg configs/fasireloc_e_outdoor.yaml --main_cfg_path JamMa/configs/jamma/outdoor/test.py --path map/megadepth