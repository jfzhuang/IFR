import os
import sys
import time
import numpy as np
import random
import argparse
import ast
from tqdm import trange
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from lib.dataset.cityscapes_video_dataset import cityscapes_video_dataset_accel, IterLoader
from lib.model.accel import Accel18
from lib.metrics import runningScore


def get_arguments():
    parser = argparse.ArgumentParser(description="Train SCNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--local_rank", type=int, help="index the replica")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--weight_res18", type=str, help="path to resnet18 pretrained weight")
    parser.add_argument("--weight_res101", type=str, help="path to resnet101 pretrained weight")
    parser.add_argument("--weight_flownet", type=str, help="path to flownet pretrained weight")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--train_num_workers", type=int)
    parser.add_argument("--test_num_workers", type=int)
    parser.add_argument("--train_iterations", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--work_dirs", type=str)

    return parser.parse_args()


def train():
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    args = get_arguments()
    if local_rank == 0:
        print(args)
        if not os.path.exists(os.path.join(args.work_dirs, args.exp_name)):
            os.makedirs(os.path.join(args.work_dirs, args.exp_name))
        tblogger = SummaryWriter(os.path.join(args.work_dirs, args.exp_name))

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        fh = logging.FileHandler(os.path.join(args.work_dirs, args.exp_name, '{}.log'.format(rq)), mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    random_seed = local_rank
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('random seed:{}'.format(random_seed))

    net = Accel18(weight_res18=args.weight_res18, weight_res101=args.weight_res101, weight_flownet=args.weight_flownet)

    params = []
    for p in net.merge.parameters():
        params.append(p)
    for p in net.flownet.parameters():
        params.append(p)
    for p in net.net_update.model.decode_head.conv_seg.parameters():
        params.append(p)
    for p in net.net_ref.model.decode_head.conv_seg.parameters():
        params.append(p)
    optimizer = optim.SGD(params=params, lr=args.lr, weight_decay=0.0005, momentum=0.9)

    net = DDP(net.cuda(), device_ids=[local_rank], output_device=local_rank)

    train_data = cityscapes_video_dataset_accel(split='train')
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.train_num_workers,
        drop_last=True,
        sampler=DistributedSampler(train_data),
        persistent_workers=True,
    )
    train_loader = IterLoader(train_loader)

    test_data = cityscapes_video_dataset_accel(split='val')
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        drop_last=False,
        persistent_workers=True,
    )

    miou_cal = runningScore(n_classes=19)
    best_miou = 0.0

    net.module.set_train()
    for step in trange(args.train_iterations):
        im_seg_list, im_flow_list, gt = next(train_loader)
        loss = net(im_seg_list, im_flow_list, gt)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_lr(args, optimizer, step)

        if local_rank == 0 and (step + 1) % args.log_interval == 0:
            # print('iter:{}/{} lr:{:.6f} loss:{:.6f}'.format(step + 1, args.train_iterations, lr, loss.item()))
            logger.info('iter:{}/{} lr:{:.6f} loss:{:.6f}'.format(step + 1, args.train_iterations, lr, loss.item()))
            tblogger.add_scalar('lr', lr, step + 1)
            tblogger.add_scalar('loss', loss.item(), step + 1)

        if (step + 1) % args.val_interval == 0:
            if local_rank == 0:
                # print('begin validation')
                logger.info('begin validation')

            net.eval()
            test_loader_iter = iter(test_loader)
            with torch.no_grad():
                for data in test_loader_iter:
                    im_seg_list, im_flow_list, gt = data
                    pred = net(im_seg_list, im_flow_list)
                    out = torch.argmax(pred, dim=1)
                    out = out.squeeze().cpu().numpy()
                    gt = gt.squeeze().cpu().numpy()
                    miou_cal.update(gt, out)
                miou = miou_cal.get_scores()
                miou_cal.reset()

            if local_rank == 0:
                tblogger.add_scalar('miou', miou, step + 1)
                if miou > best_miou:
                    best_miou = miou
                    save_path = os.path.join(args.work_dirs, args.exp_name, 'best.pth')
                    torch.save(net.state_dict(), save_path)
                # print('step:{} current miou:{:.4f} best miou:{:.4f}'.format(step + 1, miou, best_miou))
                logger.info('step:{} current miou:{:.4f} best miou:{:.4f}'.format(step + 1, miou, best_miou))

            net.module.set_train()
            dist.barrier()


def adjust_lr(args, optimizer, itr):
    warmup_itr = 1000
    warmup_lr = 0.00005

    if itr < warmup_itr:
        now_lr = warmup_lr
    else:
        now_lr = args.lr

    for group in optimizer.param_groups:
        group['lr'] = now_lr
    return now_lr


if __name__ == '__main__':
    train()
    dist.destroy_process_group()
