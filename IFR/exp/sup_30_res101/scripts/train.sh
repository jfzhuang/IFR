export PYTHONPATH=$PYTHONPATH:/home/zhuangjiafan/.local/lib/python3.8/site-packages

CONFIG_FILE=exp/sup_30_res101/configs/sup_30_res101.py
GPU_NUM=2

cd /home/zhuangjiafan/semi-vss/IFR
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}