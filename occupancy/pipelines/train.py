import argparse
import math
import os
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch import nn
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import autoencoderkl_3d, panoramic2voxel, diffusion3d

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 10
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.allow_fp16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.CUDNN_TENSOR_DTYPES.add(torch.bfloat16)
torch.manual_seed(0)
torch.random.manual_seed(0)
np.random.seed(0)


def trace_and_compile(mod: nn.Module, inputs: torch.Tensor) -> nn.Module:
    mod.eval()
    mod = torch.jit.trace(mod, inputs, check_tolerance=torch.finfo(torch.bfloat16).eps)
    mod = torch.compile(mod, fullgraph=True, dynamic=False, mode="max-autotune")
    return mod


def script_and_compile(mod: nn.Module) -> nn.Module:
    mod.eval()
    mod = torch.jit.script(mod)
    mod = torch.compile(mod, fullgraph=True, dynamic=False, mode="max-autotune")
    return mod


def remove_prefix(state_dict: dict, prefix: str) -> dict:
    noprefix_state_dict = OrderedDict()
    for key, value in state_dict.items():
        noprefix_state_dict[key.replace(prefix, "")] = value
    return noprefix_state_dict


def load_model(model, path, partial=True):
    state_dict = torch.load(path, mmap=True)
    if partial:
        partial_states = model.state_dict()
        for key in partial_states.keys():
            if partial_states[key].shape == state_dict[key].shape:
                partial_states[key] = state_dict[key]
        model.load_state_dict(partial_states, strict=False, assign=True)
    else:
        model.load_state_dict(state_dict, strict=False, assign=True)
    return model


def launch(args):
    num_gpus = torch.cuda.device_count()
    args.ddp = num_gpus > 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["WORLD_SIZE"] = str(num_gpus)
    args.world_size = num_gpus
    if args.ddp:
        mp.spawn(
            train,
            args=(args,),
            nprocs=num_gpus,
            join=True,
        )
    else:
        train(0, args)


def config_model(args: argparse.Namespace) -> nn.Module:
    match args.model:
        case "autoencoderkl":
            return autoencoderkl_3d.config_model(args)
        case "panoramic2voxel":
            return panoramic2voxel.config_model(args)
        case "diffusion3d":
            return diffusion3d.config_model(args)


def config_dataloader(args: argparse.Namespace) -> DataLoader:
    match args.model:
        case "autoencoderkl":
            return autoencoderkl_3d.config_dataloader(args)
        case "panoramic2voxel":
            return panoramic2voxel.config_dataloader(args)
        case "diffusion3d":
            return diffusion3d.config_dataloader(args)


def train(
    local_rank: int,
    args: argparse.Namespace,
):
    args.local_rank = local_rank
    print(f"[rank {local_rank}] started")
    if args.ddp:
        args.device = torch.device("cuda", local_rank)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
    model, model_input_cls = config_model(args)
    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank], gradient_as_bucket_view=True, static_graph=True)
    dl, sampler = config_dataloader(args)
    if args.ddp:
        optimizer = ZeRO(
            model.parameters(),
            AdamW,
            lr=args.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.96),
            eps=1e-8,
            parameters_as_bucket_view=True,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-5,
            betas=(0.9, 0.96),
            eps=1e-8,
        )
    total_steps = 2 * len(dl) // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    args.model_name = f"{args.model}-cls{args.num_classes}"
    with wandb.init(
        project="occupancy",
        name=f"{args.model_name}-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        config=args,
    ) as run:
        run.watch(model)
        step = 0
        for ep in range(args.total_epochs):
            args.grad_accum = args.grad_accum * (ep + 1)
            if sampler is not None:
                sampler.set_epoch(ep)
            for i, batch in enumerate(pbar := tqdm(dl, disable=args.local_rank != 0, dynamic_ncols=True)):
                model.train()
                model_input = model_input_cls(batch, dtype=args.dtype, device=args.device)

                def forward():
                    output = model(model_input)
                    mean_loss = output.loss.mean()
                    (mean_loss / args.grad_accum).backward()
                    run.log({"loss": mean_loss.item()}, step=step)
                    return output, mean_loss

                if not (step % args.grad_accum == 0 and i > 0):
                    if args.ddp:
                        with model.no_sync():
                            output, mean_loss = forward()
                    else:
                        output, mean_loss = forward()
                else:
                    output, mean_loss = forward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_description(f"Loss: {mean_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.3e}")
                if step % args.save_every == 0 and torch.isfinite(mean_loss) and local_rank == 0:
                    fig = output.figure
                    fig.savefig(f"{args.model_name}.png")
                    run.log({"output": wandb.Image(fig)}, step=step)
                    plt.close(fig)
                    model.eval()
                    state_dict = model.state_dict() if not isinstance(model, DDP) else model.module.state_dict()
                    state_dict = remove_prefix(state_dict, "_org_mod.")
                    torch.save(state_dict, os.path.join(args.save_dir, f"{args.model_name}.tmp.pt"))
                    os.rename(
                        os.path.join(args.save_dir, f"{args.model_name}.tmp.pt"),
                        os.path.join(args.save_dir, f"{args.model_name}.pt"),
                    )
                step += 1


def warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int, min_factor: float = 0.01) -> float:
    if step < warmup_steps:
        return 1e-8 + step / warmup_steps
    elif step < total_steps:
        return max(
            min_factor,
            0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / total_steps)),
        )
    else:
        return min_factor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=1)
    args = parser.parse_args()
    match args.dtype:
        case "bf16":
            args.dtype = torch.bfloat16
        case "fp32":
            args.dtype = torch.float32
        case "fp16":
            args.dtype = torch.float16
    launch(args)
