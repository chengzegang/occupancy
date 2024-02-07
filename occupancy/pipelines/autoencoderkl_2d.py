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
import torchvision.transforms.v2.functional as TF
from occupancy import ops
from occupancy.datasets.nuscenes_dataset import (
    NuScenesDataset,
    NuScenesDatasetItem,
    NuScenesDepthImage,
    NuScenesDepthImageDataset,
    NuScenesOccupancyDataset,
)
from occupancy.models.unet_2d import UnetDecoder2d, UnetEncoder2d
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


class AutoEncoderKL2dInput:
    image: Tensor
    depth: Tensor
    kl_weight: float
    clamp_min: float
    clamp_max: float

    def __init__(
        self,
        item: NuScenesDepthImage,
        kl_weight: float = 1,
        clamp_min: float = -30,
        clamp_max: float = 20,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        item = item.to(dtype=dtype, device=device)
        self.image = item.image
        self.depth = item.depth
        self.kl_weight = kl_weight
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max


class AutoEncoderKL2dOutput:
    prediction: Tensor
    ground_truth: Tensor
    image: Tensor
    kl_loss: Tensor
    latent_dist: GaussianDistribution
    latent_sample: Tensor

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
        image: Tensor,
        kl_loss: Tensor,
        latent_dist: GaussianDistribution,
        latent_sample: Tensor,
    ):
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.image = image
        self.kl_loss = kl_loss
        self.latent_dist = latent_dist
        self.latent_sample = latent_sample

    @property
    @torch.jit.ignore
    def recon_loss(self) -> Tensor:
        mask = self.ground_truth < 200
        return F.mse_loss(self.prediction, self.ground_truth, reduction="none")[mask]

    @property
    @torch.jit.ignore
    def loss(self) -> Tensor:
        return self.recon_loss.mean() + self.kl_loss.mean()

    @property
    @torch.jit.unused
    def figure(self) -> plt.Figure:
        gt = TF.to_pil_image(self.ground_truth[0].float() / 256, mode="L")
        pred = TF.to_pil_image(self.prediction[0].float() / 256, mode="L")
        img = TF.to_pil_image(self.image[0].float())
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(gt, cmap="viridis_r")
        ax.set_title("Ground Truth")
        ax.axis("off")
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(pred, cmap="viridis_r")
        ax.set_title("Prediction")
        ax.axis("off")
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(img)
        ax.set_title("Image")
        ax.axis("off")
        return fig


class AutoEncoderKL2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int = 4,
        base_channels: int = 128,
        multiplier: int = 2,
        num_layers: int = 3,
        exportable: bool = False,
    ):
        super().__init__()
        self.latent_scale = 1 / 10
        self.encoder = UnetEncoder2d(
            in_channels=in_channels,
            latent_dim=latent_dim * 2,
            base_channels=base_channels,
            multiplier=multiplier,
            num_layers=num_layers,
        )
        self.decoder = UnetDecoder2d(
            out_channels=out_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            multiplier=multiplier,
            num_layers=num_layers,
        )

    def encode_latent(self, voxel_inputs: Tensor) -> Tensor:
        latent = self.encoder(voxel_inputs)
        return latent

    def encode(self, voxel_inputs: Tensor) -> GaussianDistribution:
        latent = self.encode_latent(voxel_inputs)
        latent_dist = GaussianDistribution.from_latent(latent, latent_scale=self.latent_scale)
        return latent_dist

    def decode(self, latent: Tensor) -> Tensor:
        latent = latent / self.latent_scale
        return self.decoder(latent)

    def forward(
        self,
        input: AutoEncoderKL2dInput,
    ) -> AutoEncoderKL2dOutput:
        # NOTE: due to pytorch baddmm bug, we use pipeline optimization to force use of addmm
        latent = self.encode_latent(input.image)
        latent_dist = GaussianDistribution.from_latent(latent, input.clamp_min, input.clamp_max, self.latent_scale)
        kl_loss = input.kl_weight * latent_dist.kl_loss
        pred_output = torch.exp(self.decode(latent_dist.sample())) * 100
        return AutoEncoderKL2dOutput(
            pred_output,
            input.depth,
            input.image,
            kl_loss,
            latent_dist,
            latent_dist.sample(),
        )


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
    os.makedirs(args.save_dir, exist_ok=True)
    model = AutoEncoderKL2d(in_channels=3, out_channels=1)
    try:
        model = load_model(model, os.path.join(args.save_dir, f"{args.model}-cls{args.num_classes}.pt"), partial=True)
    except Exception as e:
        print(e)
        pass
    model.to(args.dtype).to(args.device)
    return model, AutoEncoderKL2dInput


def config_dataloader(args):
    dataset = NuScenesDepthImageDataset(args.data_dir)
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
