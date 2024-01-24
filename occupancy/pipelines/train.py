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
from torch import Tensor, nn
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import threading

from trimesh.voxel.ops import matrix_to_marching_cubes
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


def to_human_readable(num: int) -> str:
    if num < 1024:
        return f"{num}"
    elif num < 1024**2:
        return f"{num / 1024:.2f}K"
    elif num < 1024**3:
        return f"{num / 1024 ** 2:.2f}M"
    elif num < 1024**4:
        return f"{num / 1024 ** 3:.2f}B"
    else:
        return f"{num / 1024 ** 4:.2f}T"


def count_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = to_human_readable(total)
    return total


from torch.optim.lr_scheduler import LRScheduler


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-8) -> float:
    if step < warmup_steps:
        return 1e-8 + step / warmup_steps * base_lr
    elif step < total_steps:
        return max(
            min_lr,
            0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / total_steps)) * base_lr,
        )
    else:
        return min_lr


class CosineWarmupLR(LRScheduler):
    def __init__(
        self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6, last_epoch=-1, verbose=False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            cosine_warmup_lr(self._step_count, self.warmup_steps, self.total_steps, base_lr, self.min_lr)
            for base_lr in self.base_lrs
        ]


def save_state_dict(state_dict: dict, path: str):
    torch.save(state_dict, path + ".tmp")
    os.rename(path + ".tmp", path)


def record(run, output, model: nn.Module, args: argparse.Namespace, step: int):
    model.eval()
    state_dict = model.state_dict() if not isinstance(model, DDP) else model.module.state_dict()
    save_state_dict(state_dict, os.path.join(args.save_dir, f"{args.model_name}.pt"))
    fig = output.figure
    fig.savefig(f"{args.model_name}.png")
    run.log({"output": wandb.Image(fig)}, step=step)
    plt.close(fig)
    prediction = output.prediction[0, 0] > 0
    prediction = prediction.cpu().permute(1, 2, 0)
    ground_truth = output.ground_truth[0, 0] > 0
    ground_truth = ground_truth.cpu().permute(1, 2, 0)
    if prediction.sum() > 0:
        save_as_obj(prediction, f"{args.model_name}_prediction.obj")
        save_as_obj(ground_truth, f"{args.model_name}_ground_truth.obj")
        run.log({"prediction": wandb.Object3D(f"{args.model_name}_prediction.obj")}, step=step)
        run.log({"ground_truth": wandb.Object3D(f"{args.model_name}_ground_truth.obj")}, step=step)


def save_as_obj(voxel: Tensor, path: str):
    voxel = voxel.numpy()
    mesh = matrix_to_marching_cubes(voxel)
    mesh.export(path)


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
    total_params = count_parameters(model)
    print(f"[rank {local_rank}] total params: {total_params}")
    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank], gradient_as_bucket_view=True, static_graph=True)
    dl, sampler = config_dataloader(args)
    if args.ddp:
        optimizer = ZeRO(
            model.parameters(),
            AdamW,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            parameters_as_bucket_view=True,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    total_steps = 2 * len(dl) // args.grad_accum
    scheduler = CosineWarmupLR(optimizer, args.warmup_steps, total_steps)
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
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_description(f"Loss: {mean_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.3e}")
                if step % args.save_every == 0 and torch.isfinite(mean_loss) and local_rank == 0:
                    record(run, output, model, args, step)
                step += 1


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
    parser.add_argument("--warmup-steps", type=int, default=0)

    args = parser.parse_args()
    match args.dtype:
        case "bf16":
            args.dtype = torch.bfloat16
        case "fp32":
            args.dtype = torch.float32
        case "fp16":
            args.dtype = torch.float16
    launch(args)
