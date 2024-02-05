import argparse
import math
import os
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from tensordict import tensorclass
from torch import Tensor, nn
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.quantization import DeQuantStub, QuantStub, fuse_modules
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from occupancy import ops
from occupancy.datasets.nuscenes_dataset import (
    NuScenesDataset,
    NuScenesDatasetItem,
    NuScenesMixOccupancyDataset,
    NuScenesOccupancyDataset,
)
from occupancy.models.transformer import Transformer
from occupancy.models.unet_3d import UnetDecoder3d, UnetEncoder3d


@tensorclass
class GaussianDistribution:
    latent: Tensor
    mean: Tensor
    logvar: Tensor
    latent_scale: float

    @classmethod
    def from_latent(
        cls, latent: Tensor, clamp_min: float = -30, clamp_max: float = 20, latent_scale: float = 1.0
    ) -> "GaussianDistribution":
        mean, logvar = torch.chunk(latent, 2, dim=1)
        logvar = torch.clamp(logvar, clamp_min, clamp_max)
        return cls(latent, mean, logvar, latent_scale, batch_size=[latent.shape[0]])

    def sample(self) -> Tensor:
        return (self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)) * self.latent_scale

    @property
    def kl_loss(self) -> Tensor:
        return -0.5 * (1 + self.logvar - self.mean**2 - self.logvar.exp()).mean(dim=-1)

    def kl_div(self, other: "GaussianDistribution") -> Tensor:
        return 0.5 * (
            other.logvar
            - self.logvar
            + (self.logvar.exp() + (self.mean - other.mean) ** 2) / (other.logvar.exp() + 1e-8)
            - 1
        )

    def nll(self, sample: Tensor) -> Tensor:
        return 0.5 * ((sample - self.mean) ** 2 / (self.logvar.exp() + 1e-8) + self.logvar)


class VoxelAugmentation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, voxel: Tensor) -> Tensor:
        if self.training:
            voxel = voxel + (torch.rand_like(voxel) > 0.996).type_as(voxel)
            return voxel


class LatentEncoder2dInput:
    occupancy: Tensor
    kl_weight: float = 0.00001
    clamp_min: float = -30
    clamp_max: float = 20

    def __init__(
        self,
        voxel: Tensor,
        kl_weight: float = 0.00001,
        clamp_min: float = -30,
        clamp_max: float = 20,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        voxel = voxel.type(dtype).to(device)
        self.occupancy = voxel  # self.splicing(voxel)
        self.kl_weight = kl_weight
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def splicing(self, voxel: Tensor):
        i = random.randint(0, 255)
        j = random.randint(0, 255)
        return voxel[..., i : i + 256, j : j + 256, :]


class LatentEncoder2dOutput:
    prediction: Tensor
    ground_truth: Tensor
    loss: Tensor

    _CMAP = np.asarray(
        [  # RGB.
            (np.nan, np.nan, np.nan),  # None
            (0, 0, 0),  # Black. noise
            (112, 128, 144),  # Slategrey barrier
            (220, 20, 60),  # Crimson bicycle
            (255, 127, 80),  # Orangered bus
            (255, 158, 0),  # Orange car
            (233, 150, 70),  # Darksalmon construction
            (255, 61, 99),  # Red motorcycle
            (0, 0, 230),  # Blue pedestrian
            (47, 79, 79),  # Darkslategrey trafficcone
            (255, 140, 0),  # Darkorange trailer
            (255, 99, 71),  # Tomato truck
            (0, 207, 191),  # nuTonomy green driveable_surface
            (175, 0, 75),  # flat other
            (75, 0, 75),  # sidewalk
            (112, 180, 60),  # terrain
            (222, 184, 135),  # Burlywood mannade
            (0, 175, 0),  # Green vegetation
        ]
    )

    def __init__(
        self,
        prediction: Tensor,
        ground_truth: Tensor,
        kl_loss: Tensor,
        latent_dist: GaussianDistribution,
        latent_sample: Tensor,
    ):
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.kl_loss = kl_loss
        self.latent_dist = latent_dist
        self.latent_sample = latent_sample

    @property
    @torch.jit.ignore
    def recon_loss(self) -> Tensor:
        if self.ground_truth.size(1) == 1:
            return F.binary_cross_entropy_with_logits(
                self.prediction,
                self.ground_truth,
                reduction="none",
                pos_weight=torch.tensor(3.2, device=self.prediction.device),
            )
        label = self.ground_truth.argmax(dim=1)
        population = torch.bincount(label.flatten(), minlength=18).float()
        weight = torch.pow(torch.numel(label) / population * 4 * math.pi / 3, 1 / 3)
        loss = F.cross_entropy(
            self.prediction.permute(0, 2, 3, 4, 1).flatten(0, -2),
            label.flatten(),
            weight=weight.type_as(self.prediction),
        )
        return loss

    @property
    @torch.jit.ignore
    def loss(self) -> Tensor:
        return self.recon_loss.mean() + self.kl_loss.mean()

    @property
    @torch.jit.unused
    def iou(self):
        if self.ground_truth.size(1) == 1:
            return ops.iou(self.prediction.flatten() >= 0, self.ground_truth.flatten() > 0, 2, 0)
        else:
            return ops.iou(
                self.prediction.argmax(dim=1).flatten() > 0, self.ground_truth.argmax(dim=1).flatten() > 0, 2, 0
            )

    @property
    @torch.jit.unused
    def figure(self) -> plt.Figure:
        i, j, k = None, None, None
        ih, jh, kh = None, None, None
        c = None
        ch = None
        if self.ground_truth.size(1) == 1:
            i, j, k = torch.where(self.ground_truth[0, 0].detach().cpu() > 0)
            ih, jh, kh = torch.where(self.prediction[0, 0].detach().cpu() >= 0)
            c = k
            ch = kh
        else:
            label = self.ground_truth[0].argmax(dim=0).detach().cpu()
            i, j, k = torch.where(label)
            c = label[i, j, k]
            c = self._CMAP[c] / 255.0

            pred = self.prediction[0].argmax(dim=0).detach().cpu()
            ih, jh, kh = torch.where(pred)
            ch = pred[ih, jh, kh]
            ch = self._CMAP[ch] / 255.0

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"iou: {self.iou[0, 1].item():.2%}")
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(i, j, k, c=c, s=1, marker="s")
        ax.set_title("Ground Truth")
        ax.set_box_aspect((1, 1, self.ground_truth.shape[-1] / self.ground_truth.shape[-3]))
        ax.set_xlim(0, self.ground_truth.shape[-3])
        ax.set_ylim(0, self.ground_truth.shape[-2])
        ax.set_zlim(0, self.ground_truth.shape[-1])
        ax.set_zticks([])

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(ih, jh, kh, c=ch, s=1, marker="s")
        ax.set_title("Prediction")
        ax.set_box_aspect((1, 1, self.ground_truth.shape[-1] / self.ground_truth.shape[-3]))
        ax.set_xlim(0, self.ground_truth.shape[-3])
        ax.set_ylim(0, self.ground_truth.shape[-2])
        ax.set_zlim(0, self.ground_truth.shape[-1])
        ax.set_zticks([])

        return fig


@dataclass
class AutoEncoderKL3dConfig:
    in_channels: int = 1
    out_channels: int = 1
    latent_dim: int = 64
    base_channels: int = 64
    multiplier: int = 2
    num_layers: int = 4
    num_attention_layers: int = 3


class LatentEncoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        num_attention_layers: int = 3,
        exportable: bool = False,
    ):
        super().__init__()
        self.latent_scale = 1 / 10
        self.voxel_augmentation = VoxelAugmentation()
        from diffusers import AutoencoderKL

        self.image_autoencoderkl = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae", torch_dtype=torch.bfloat16, torchscript=True, device_map="auto"
        )
        self.encoder = Transformer(self.hidden_size, 8, self.hidden_size // 128, 128, max_seq_length=10000)

    def forward(
        self,
        input: LatentEncoder2dInput,
    ) -> LatentEncoder2dOutput:
        # NOTE: due to pytorch baddmm bug, we use pipeline optimization to force use of addmm
        image = input.image
        with torch.no_grad():
            latent = self.image_autoencoderkl.encode(image).latent_dist.sample()
        latent = self.encoder(latent.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(latent)
        pred = self.image_autoencoderkl.decode(latent)
        loss = F.mse_loss(pred, input.target)
        return LatentEncoder2dOutput(
            pred,
            image,
            loss,
        )

    @classmethod
    def from_config(cls, config: AutoEncoderKL3dConfig) -> "LatentEncoder2d":
        return cls(**asdict(config))


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


def config_model(args):
    model_config = AutoEncoderKL3dConfig(in_channels=args.num_classes, out_channels=args.num_classes)
    os.makedirs(args.save_dir, exist_ok=True)
    model = LatentEncoder2d.from_config(model_config)
    try:
        model = load_model(model, os.path.join(args.save_dir, f"autoencoderkl-cls{args.num_classes}.pt"), partial=True)
    except Exception as e:
        print(e)
        pass
    model.to(args.dtype).to(args.device)
    return model, LatentEncoder2dInput


def config_dataloader(args):
    dataset = None
    if args.num_classes == 1:
        dataset = NuScenesMixOccupancyDataset(args.data_dir)
    else:
        dataset = NuScenesOccupancyDataset(args.data_dir, binary=args.num_classes == 1)
    sampler = None
    if args.ddp:
        sampler = DistributedSampler(dataset)
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
    )
    return dl, sampler
