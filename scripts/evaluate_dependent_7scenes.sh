export PYTHONPATH="JamMa:$PYTHONPATH"

python fasireloc_e.py -m map/7scenes/chess --cfg configs/fasireloc_e_indoor.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py 
python fasireloc_e.py -m map/7scenes/fire --cfg configs/fasireloc_e_indoor.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py 
python fasireloc_e.py -m map/7scenes/heads --cfg configs/fasireloc_e_indoor.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py
python fasireloc_e.py -m map/7scenes/office --cfg configs/fasireloc_e_indoor.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py
python fasireloc_e.py -m map/7scenes/pumpkin --cfg configs/fasireloc_e_indoor.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py 
python fasireloc_e.py -m map/7scenes/redkitchen --cfg configs/fasireloc_e_indoor.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py 
python fasireloc_e.py -m map/7scenes/stairs --cfg configs/fasireloc_e_indoor.yaml --main_cfg_path JamMa/configs/jamma/indoor/test.py 
 