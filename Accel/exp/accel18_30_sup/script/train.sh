#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/zhuangjiafan/anaconda3/envs/pytorch1.8/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/home/zhuangjiafan/semi-vss/mmsegmentation-0.14.0
export PYTHONPATH=$PYTHONPATH:/home/zhuangjiafan/semi-vss/release/Accel

cd /home/zhuangjiafan/semi-vss/release/Accel && \
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM exp/accel18_30_sup/python/train.py \
        --exp_name accel18_30_sup \
        --weight_res18 ./pretrained/sup_30_res18/best_mIoU_iter_31000.pth \
        --weight_res101 ./pretrained/sup_30_res101/best_mIoU_iter_39000.pth \
        --weight_flownet ./pretrained/flownet_pretrained.pth \
        --lr 0.0005 \
        --train_batch_size 2 \
        --train_num_workers 2 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --train_iterations 40000 \
        --log_interval 50 \
        --val_interval 1000 \
        --work_dirs /home/zhuangjiafan/semi-vss/release/Accel/work_dirs \