import cv2
import os
import argparse
import numpy as np
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
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


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def reduce_metrics(metrics, device, world_size):
    if world_size == 1:
        return metrics

    keys = list(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device)
    dist.reduce(values, dst=0, op=dist.ReduceOp.SUM)
    values /= world_size
    return dict(zip(keys, values.cpu().tolist()))


def train(args):
    """ main training loop """

    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    distributed = world_size > 1
    logger = None

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    datapath = args.datapath
    if datapath is None:
        datapath = "datasets/TartanAir" if args.dataset == "tartan" else "datasets/tartanair-v2"
    db = dataset_factory([args.dataset], datapath=datapath, n_frames=args.n_frames)
    sampler = DistributedSampler(
        db, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    ) if distributed else None
    train_loader = DataLoader(
        db, batch_size=1, shuffle=sampler is None, sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0)

    model = VONet().to(device)

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict, strict=False)

    net = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False
    ) if distributed else model
    net.train()

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    if rank == 0:
        Path("checkpoints").mkdir(exist_ok=True)
        logger = Logger(args.name, scheduler)
        print(
            f"Training on {world_size} GPU(s) with global batch size {world_size} "
            f"and {len(db)} samples"
        )

    total_steps = 0
    epoch = 0

    try:
        while total_steps < args.steps:
            if sampler is not None:
                sampler.set_epoch(epoch)

            for data_blob in train_loader:
                images, poses, disps, intrinsics = [
                    x.to(device, non_blocking=True).float() for x in data_blob
                ]
                optimizer.zero_grad()

                # fix poses to gt for first 1k steps
                so = total_steps < 1000 and args.ckpt is None

                poses = SE3(poses).inv()
                traj = net(
                    images, poses, disps, intrinsics,
                    M=1024, STEPS=18, structure_only=so)

                loss = 0.0
                for i, (v, x, y, P1, P2, kl) in enumerate(traj):
                    e = (x - y).norm(dim=-1)
                    e = e.reshape(-1, model.P**2)[
                        (v > 0.5).reshape(-1)].min(dim=-1).values

                    N = P1.shape[1]
                    indices = torch.arange(N, device=device)
                    ii, jj = torch.meshgrid(indices, indices, indexing="ij")
                    ii = ii.reshape(-1)
                    jj = jj.reshape(-1)

                    k = ii != jj
                    ii = ii[k]
                    jj = jj[k]

                    P1 = P1.inv()
                    P2 = P2.inv()

                    t1 = P1.matrix()[..., :3, 3]
                    t2 = P2.matrix()[..., :3, 3]

                    s = kabsch_umeyama(t2[0], t1[0]).detach().clamp(max=10.0)
                    P1 = P1.scale(s.view(1, 1))

                    dP = P1[:, ii].inv() * P1[:, jj]
                    dG = P2[:, ii].inv() * P2[:, jj]

                    e1 = (dP * dG.inv()).log()
                    tr = e1[..., 0:3].norm(dim=-1)
                    ro = e1[..., 3:6].norm(dim=-1)

                    loss += args.flow_weight * e.mean()
                    if not so and i >= 2:
                        loss += args.pose_weight * (tr.mean() + ro.mean())

                # kl is 0 (no longer used)
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

                metrics = reduce_metrics(metrics, device, world_size)
                if rank == 0:
                    logger.push(metrics)

                checkpoint_step = (
                    total_steps % args.checkpoint_freq == 0
                    or total_steps == args.steps
                )
                if checkpoint_step:
                    torch.cuda.empty_cache()

                    if rank == 0:
                        path = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                        torch.save(model.state_dict(), path)

                    if distributed:
                        dist.barrier()

                    if not args.skip_validation:
                        if rank == 0:
                            from evaluate_tartan import evaluate as validate
                            validation_results = validate(None, model)
                            logger.write_dict(validation_results)

                        if distributed:
                            dist.barrier()

                    torch.cuda.empty_cache()
                    net.train()

                if total_steps >= args.steps:
                    break

            epoch += 1
    finally:
        if logger is not None:
            logger.close()
        if dist.is_initialized():
            dist.destroy_process_group()


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
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--pose_weight', type=float, default=10.0)
    parser.add_argument('--flow_weight', type=float, default=0.1)
    args = parser.parse_args()

    train(args)
