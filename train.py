import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dpvo.data_readers.factory import dataset_factory

from dpvo.lietorch import SE3
from dpvo.logger import Logger
import torch.nn.functional as F

from dpvo.net import VONet

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def image2gray(image):
    image = image.mean(dim=0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1)**2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def train(args):
    """ main training loop """

    # legacy ddp code
    rank = 0

    datapath = args.datapath
    if datapath is None:
        datapath = "datasets/TartanAir" if args.dataset == "tartan" else "datasets/tartanair-v2"
    db = dataset_factory([args.dataset], datapath=datapath, n_frames=args.n_frames)
    train_loader = DataLoader(
        db, batch_size=1, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0)

    net = VONet()
    net.train()
    net.cuda()

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict, strict=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if rank == 0:
        Path("checkpoints").mkdir(exist_ok=True)
        logger = Logger(args.name, scheduler)

    total_steps = 0

    while 1:
        for data_blob in train_loader:
            images, poses, disps, intrinsics = [x.cuda().float() for x in data_blob]
            optimizer.zero_grad()

            # fix poses to gt for first 1k steps
            so = total_steps < 1000 and args.ckpt is None

            poses = SE3(poses).inv()
            traj = net(images, poses, disps, intrinsics, M=1024, STEPS=18, structure_only=so)

            loss = 0.0
            for i, (v, x, y, P1, P2, kl) in enumerate(traj):
                e = (x - y).norm(dim=-1)
                e = e.reshape(-1, net.P**2)[(v > 0.5).reshape(-1)].min(dim=-1).values

                N = P1.shape[1]
                ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
                ii = ii.reshape(-1).cuda()
                jj = jj.reshape(-1).cuda()

                k = ii != jj
                ii = ii[k]
                jj = jj[k]

                P1 = P1.inv()
                P2 = P2.inv()

                t1 = P1.matrix()[...,:3,3]
                t2 = P2.matrix()[...,:3,3]

                s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                P1 = P1.scale(s.view(1, 1))

                dP = P1[:,ii].inv() * P1[:,jj]
                dG = P2[:,ii].inv() * P2[:,jj]

                e1 = (dP * dG.inv()).log()
                tr = e1[...,0:3].norm(dim=-1)
                ro = e1[...,3:6].norm(dim=-1)

                loss += args.flow_weight * e.mean()
                if not so and i >= 2:
                    loss += args.pose_weight * ( tr.mean() + ro.mean() )

            # kl is 0 (not longer used)
            loss += kl
            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            metrics = {
                "loss": loss.item(),
                "kl": kl.item(),
                "px1": (e < .25).float().mean().item(),
                "ro": ro.float().mean().item(),
                "tr": tr.float().mean().item(),
                "r1": (ro < .001).float().mean().item(),
                "r2": (ro < .01).float().mean().item(),
                "t1": (tr < .001).float().mean().item(),
                "t2": (tr < .01).float().mean().item(),
            }

            if rank == 0:
                logger.push(metrics)

            checkpoint_step = total_steps % args.checkpoint_freq == 0 or total_steps == args.steps
            if checkpoint_step:
                torch.cuda.empty_cache()

                if rank == 0:
                    PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                    torch.save(net.state_dict(), PATH)

                if not args.skip_validation:
                    from evaluate_tartan import evaluate as validate
                    validation_results = validate(None, net)
                    if rank == 0:
                        logger.write_dict(validation_results)

                torch.cuda.empty_cache()
                net.train()

            if total_steps >= args.steps:
                if rank == 0:
                    logger.close()
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--dataset', choices=['tartan', 'tartan_v2'], default='tartan')
    parser.add_argument('--datapath', help='dataset root (defaults to the repository dataset path)')
    parser.add_argument('--steps', type=int, default=240000)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--clip', type=float, default=10.0)
    parser.add_argument('--n_frames', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_freq', type=int, default=10000)
    parser.add_argument('--skip_validation', action='store_true')
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    args = parser.parse_args()

    train(args)
