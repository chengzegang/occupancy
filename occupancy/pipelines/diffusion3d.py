import argparse
import math
import os
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Tuple

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
from occupancy.datasets.nuscenes import NuScenesDataset, NuScenesDatasetItem, NuScenesImage, NuScenesOccupancyDataset
from occupancy.models.unet_3d import UnetDecoder3d, UnetEncoder3d
from occupancy.models.unet_conditional_attention_timestep_3d import UnetConditionalAttention3d
from occupancy.pipelines.panoramic2voxel import VisionTransformerFeatureExtractor
from .autoencoderkl_3d import (
    AutoEncoderKL3dInput,
    AutoEncoderKL3dOutput,
    AutoEncoderKL3dConfig,
    AutoEncoderKL3d,
    GaussianDistribution,
)
from torch.utils.data import Subset
from diffusers import AutoencoderKL, DDIMScheduler


class Diffusion3dInput:
    images: Tensor
    voxel: Tensor

    def __init__(self, images: Tensor, voxel: Tensor):
        self.images = images
        self.voxel = voxel

    @classmethod
    def from_nuscenes_dataset_item(
        cls,
        batch: NuScenesDatasetItem,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "Diffusion3dInput":
        images: NuScenesImage = torch.stack(
            [
                batch.cam_front_left,
                batch.cam_front,
                batch.cam_front_right,
                batch.cam_back_right,
                batch.cam_back,
                batch.cam_back_left,
            ],
            dim=1,
        )
        images.data = images.data.to(device=device, dtype=dtype, non_blocking=True)
        images.intrinsic = images.intrinsic.to(device=device, dtype=dtype, non_blocking=True)
        images.rotation = images.rotation.to(device=device, dtype=dtype, non_blocking=True)
        images.translation = images.translation.to(device=device, dtype=dtype, non_blocking=True)
        batch.lidar_top.location = batch.lidar_top.location.to(device=device, dtype=dtype, non_blocking=True)
        batch.lidar_top.attribute = batch.lidar_top.attribute.to(device=device, non_blocking=True)

        voxel = batch.lidar_top.voxel.to(device=device, dtype=dtype, non_blocking=True)
        model_input = cls(
            images=images.data,
            voxel=voxel,
        )
        return model_input


class Diffusion3dOutput:
    prediction: Tensor
    ground_truth: Tensor
    loss: Optional[Tensor] = None

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
        loss: Optional[Tensor] = None,
    ):
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.loss = loss

    @property
    @torch.jit.unused
    def iou(self):
        return ops.iou(self.prediction.flatten() >= 0, self.ground_truth.flatten() > 0, 2, 0)

    @property
    @torch.jit.unused
    def figure(self) -> plt.Figure:
        ground_truth = self.ground_truth
        i, j, k = torch.where(ground_truth[0, 0].detach().cpu() > 0)
        c = ground_truth[0, 0, i, j, k].detach().cpu()
        c = k
        prediction = self.prediction
        ih, jh, kh = torch.where(prediction[0, 0].detach().cpu() >= 0)
        ch = prediction[0, 0, ih, jh, kh].detach().cpu()
        ch = kh
        # ch = self.prediction.data[0, ih, jh, kh]
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"iou: {self.iou[0, 1].item():.2%}")
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(i, j, k, c=c, s=1, marker="s", alpha=1)
        ax.set_title("Ground Truth")
        # ax.set_box_aspect((1, 1, 1 / 4))
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1 / 10))
        ax.view_init(azim=-60, elev=45)

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(ih, jh, kh, c=ch, s=1, marker="s", alpha=1)
        ax.set_title("Prediction")
        # ax.set_box_aspect((1, 1, 1 / 4))
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1 / 10))
        ax.view_init(azim=-60, elev=45)

        return fig


class Diffusion3d(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_autoencoderkl_model_id: str = "stabilityai/sdxl-vae",
    ):
        super().__init__()
        self.latent_scale = 0.125
        self.num_classes = num_classes
        self.voxel_encoder_latent_dim = 16
        self.plane2polar_depth_channels = 16
        self.image_autoencoderkl = AutoencoderKL.from_pretrained(
            image_autoencoderkl_model_id, torch_dtype=torch.bfloat16, torchscript=True, device_map="auto"
        )
        self.voxel_autoencoderkl = AutoEncoderKL3d(
            num_classes,
            num_classes,
            16,
            64,
            2,
            3,
        )
        self.image_encoder = VisionTransformerFeatureExtractor(
            self.image_autoencoderkl.config.latent_channels, 128, 512, 2, 6, 128
        )
        self.decoder = UnetConditionalAttention3d(
            self.voxel_encoder_latent_dim,
            self.voxel_encoder_latent_dim,
            512,
            512,
            128,
            2,
            2,
            4,
            head_size=128,
        )

        self.scheduler = DDIMScheduler()
        self._vae_slicing = False

    def enable_vae_slicing(self, enabled: bool = True):
        self._vae_slicing = enabled

    def prepare_image(self, images: Tensor) -> Tensor:
        if not self._vae_slicing:
            return self.image_autoencoderkl.encode(images).latent_dist.sample()
        image_chunks = torch.split(images, 512, dim=-1)
        image_chunks = [torch.split(chunk, 512, dim=-2) for chunk in image_chunks]
        image_sample = []
        for i, chunks in enumerate(image_chunks):
            image_sample.append([])
            for j, chk in enumerate(chunks):
                chk = self.image_autoencoderkl.encode(chk).latent_dist.sample()
                image_sample[i].append(chk)
        image_sample = torch.cat([torch.cat(chk, dim=-2) for chk in image_sample], dim=-1)
        return image_sample.detach()

    def __call__(self, input: Diffusion3dInput) -> Diffusion3dOutput:
        return super().__call__(input)

    def prepare_latent_dist(self, voxel: Tensor) -> GaussianDistribution:
        voxel_chunks = torch.split(voxel, 256, dim=-2)
        voxel_chunks = [torch.split(chunk, 256, dim=-3) for chunk in voxel_chunks]
        voxel_dist = torch.zeros(
            voxel.shape[0],
            32,
            len(voxel_chunks[0]) * 32,
            len(voxel_chunks) * 32,
            8,
            device=voxel.device,
            dtype=voxel.dtype,
        )
        for i, chunk in enumerate(voxel_chunks):
            for j, chk in enumerate(chunk):
                chk = self.voxel_autoencoderkl.encode_latent(chk)
                voxel_dist[:, :, j * 32 : (j + 1) * 32, i * 32 : (i + 1) * 32].copy_(chk, non_blocking=True)
        voxel_dist = GaussianDistribution.from_latent(
            voxel_dist.detach(), latent_scale=self.voxel_autoencoderkl.latent_scale
        )
        return voxel_dist

    def decode_latent_sample(self, latent_sample: Tensor, out_shape: Tuple[int, int, int]) -> Tensor:
        sample = torch.zeros(
            latent_sample.shape[0],
            self.num_classes,
            *out_shape,
            device=latent_sample.device,
            dtype=latent_sample.dtype,
        )
        if not self._vae_slicing:
            sample = self.voxel_autoencoderkl.decode(latent_sample)
        else:
            latent_chunks = torch.split(latent_sample, 32, dim=-2)
            latent_chunks = [torch.split(chunk, 32, dim=-3) for chunk in latent_chunks]
            for i, chunk in enumerate(latent_chunks):
                for j, chk in enumerate(chunk):
                    chk = self.voxel_autoencoderkl.decode(chk)
                    sample[:, :, j * 256 : (j + 1) * 256, i * 256 : (i + 1) * 256].copy_(chk, non_blocking=True)

        return sample

    def forward(self, input: Diffusion3dInput) -> Diffusion3dOutput:
        batch_size = input.images.shape[0]
        num_views = input.images.shape[1]
        multiview_sample = self.prepare_image(input.images.flatten(0, 1))
        multiview_sample = self.image_encoder(multiview_sample)
        multiview_sample = multiview_sample.view(batch_size, num_views, *multiview_sample.shape[1:])
        multiview_sample = torch.cat(multiview_sample.unbind(1), dim=-1)

        with torch.no_grad():
            voxel_latent = self.prepare_latent_dist(input.voxel).sample()

        timestep_ind = random.randint(0, len(self.scheduler.timesteps) - 1)
        timestep = self.scheduler.timesteps[timestep_ind]
        noises = torch.randn_like(voxel_latent)
        noised = self.scheduler.add_noise(voxel_latent, noises, timestep)
        model_input = self.scheduler.scale_model_input(noised, timestep)
        model_output = self.decoder(
            model_input, multiview_sample, timestep.view(1, 1).expand(batch_size, -1).type_as(model_input)
        )
        loss = F.mse_loss(model_output, noises)
        voxel = input.voxel
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].type_as(model_input)
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (model_input - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        with torch.no_grad():
            pred_voxel = self.decode_latent_sample(pred_original_sample, input.voxel.shape[-3:])

        return Diffusion3dOutput(
            pred_voxel,
            voxel,
            loss,
        )

    def state_dict(self, *, prefix: str = "", keep_vars: bool = False) -> dict:
        state_dict = {
            "image_encoder": self.image_encoder.state_dict(None, prefix, keep_vars),
            "decoder": self.decoder.state_dict(None, prefix, keep_vars),
        }
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.image_encoder.load_state_dict(state_dict["image_encoder"], strict, assign)
        self.decoder.load_state_dict(state_dict["decoder"], strict, assign)


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


def check_bias(model: nn.Module):
    for mod in model.modules():
        if hasattr(mod, "bias") and mod.bias is not None:
            raise RuntimeError(f"bias found: {mod}")


def config_model(args):
    image_autoencoderkl_model_id = os.path.join(args.save_dir, "image_autoencoderkl")
    model = None
    if os.path.exists(image_autoencoderkl_model_id):
        model = Diffusion3d(num_classes=args.num_classes, image_autoencoderkl_model_id=image_autoencoderkl_model_id)

    else:
        model = Diffusion3d(num_classes=args.num_classes)
        model.image_autoencoderkl.save_pretrained(image_autoencoderkl_model_id)
    try:
        path = os.path.join(args.save_dir, f"diffusion3d-cls{args.num_classes}.pt")
        model.load_state_dict(
            torch.load(
                path,
                mmap=True,
            ),
            assign=True,
        )
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
    model.voxel_autoencoderkl.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, f"autoencoderkl-cls{args.num_classes}.pt"),
            mmap=True,
        ),
        assign=True,
    )
    check_bias(model.voxel_autoencoderkl)
    check_bias(model.image_encoder)
    check_bias(model.decoder)
    model.to(dtype=args.dtype, device=args.device, non_blocking=True)
    model.voxel_autoencoderkl.encoder = torch.jit.script(model.voxel_autoencoderkl.encoder)
    model.voxel_autoencoderkl.decoder = torch.jit.script(model.voxel_autoencoderkl.decoder)
    model.voxel_autoencoderkl.requires_grad_(False)
    model.image_autoencoderkl.requires_grad_(False)
    return model, Diffusion3dInput.from_nuscenes_dataset_item


def config_dataloader(args):
    dataset = NuScenesDataset(args.data_dir)
    index = list(range(len(dataset)))[2000:]
    dataset = Subset(dataset, index)
    sampler = DistributedSampler(dataset) if args.ddp else None
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False if args.ddp else True,
        sampler=sampler if args.ddp else None,
    )
    return dl, sampler
