from collections import OrderedDict
from datetime import datetime
from itertools import chain
import random
from typing import Any, Mapping, Optional, Tuple, overload
from dataclasses import asdict, dataclass
from functools import cached_property, partial, wraps
import math
import os
import numpy as np
import torch
from torch import nn, Tensor
import yaml
import argparse
import torch.nn.functional as F
from tensordict import tensorclass
from occupancy.datasets.nuscenes import (
    NuScenesDatasetItem,
    NuScenesImage,
    NuScenesDataset,
)
import torch.multiprocessing as mp
import torch.distributed as dist
from occupancy.models.transformer import Transformer
from occupancy.models.unet_attention_2d import UnetAttention2d
from occupancy.models.unet_attention_3d import UnetAttention3d
from occupancy.models.unet_conditional_attention_3d import UnetConditionalAttention3d

from occupancy import ops
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from occupancy.models.unet_2d import SpatialRMSNorm, Unet2d, UnetEncoder2d
from occupancy.models.unet_3d import UnetEncoder3d, UnetLatentAttention3d
from .autoencoderkl_3d import AutoEncoderKL3d, GaussianDistribution
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import InterpolationMode
from torch.utils.data import Subset
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class RandomNoise(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * 0.98 + torch.randn_like(x) * 0.02


class ImageAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = T.Compose(
            [
                T.RandomAffine(
                    degrees=(-15, 15), translate=(0.02, 0.15), shear=(-5, 5), interpolation=InterpolationMode.BILINEAR
                ),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.01),
                T.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.1, 5), value=0, inplace=False),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 5.0)),
                RandomNoise(),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)


class MultiViewImageToVoxelPipelineInput:
    images: Tensor
    voxel: Tensor
    occupancy: Tensor

    def __init__(self, images: Tensor, voxel: Tensor, occupancy: Tensor):
        self.images = images
        self.voxel = voxel
        self.occupancy = occupancy

    @classmethod
    def from_nuscenes_dataset_item(
        cls,
        batch: NuScenesDatasetItem,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "MultiViewImageToVoxelPipelineInput":
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
        occupancy = batch.lidar_top.occupancy.to(device=device, dtype=dtype, non_blocking=True)
        voxel = batch.lidar_top.voxel.to(device=device, dtype=dtype, non_blocking=True)
        model_input = cls(
            images=images.data,
            voxel=voxel,
            occupancy=occupancy,
        )
        return model_input


class MultiViewImageToVoxelPipelineOutput:
    prediction: Tensor
    ground_truth: Tensor
    occupancy: Tensor
    images: Tensor
    loss: Optional[Tensor] = None
    pos_weight: Optional[Tensor] = None

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
        occupancy: Tensor,
        images: Tensor,
        loss: Optional[Tensor] = None,
        pos_weight=None,
    ):
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.occupancy = occupancy
        self.images = images
        self.loss = loss
        self.pos_weight = pos_weight

    @property
    @torch.jit.unused
    def iou(self):
        return ops.iou(self.prediction.flatten() >= 0, self.ground_truth.flatten() > 0, 2, 0)

    @property
    @torch.jit.unused
    def figure(self) -> plt.Figure:
        occupancy = self.occupancy
        oi, oj, ok = torch.where(occupancy[0, 0].detach().cpu() > 0)
        oc = occupancy[0, 0, oi, oj, ok].detach().cpu()
        oc = ok

        ground_truth = self.ground_truth
        i, j, k = torch.where(ground_truth[0, 0].detach().cpu() > 0)
        c = ground_truth[0, 0, i, j, k].detach().cpu()
        c = k

        prediction = self.prediction
        ih, jh, kh = torch.where(prediction[0, 0].detach().cpu() >= 0)
        ch = prediction[0, 0, ih, jh, kh].detach().cpu()
        ch = kh

        x_size = self.prediction.size(-3)
        y_size = self.prediction.size(-2)

        fig = plt.figure(figsize=(30, 10))
        fig.suptitle(f"iou: {self.iou[0, 1].item():.2%}")

        ax = fig.add_subplot(1, 3, 1, projection="3d")
        ax.scatter(oi, oj, ok, c=oc, s=1, marker="s", alpha=1)
        ax.set_title("Full Occupancy")

        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1 / 10))
        ax.view_init(azim=-60, elev=30)

        ax = fig.add_subplot(1, 3, 2, projection="3d")
        ax.scatter(i, j, k, c=c, s=1, marker="s", alpha=1)
        ax.set_title("Observable Occupancy")

        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1 / 10))
        ax.view_init(azim=-60, elev=30)

        ax = fig.add_subplot(1, 3, 3, projection="3d")
        ax.scatter(ih, jh, kh, c=ch, s=1, marker="s", alpha=1)
        ax.set_title("Prediction")

        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1 / 10))
        ax.view_init(azim=-60, elev=30)

        return fig


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


@dataclass
class MultiViewImageToVoxelConfig:
    @classmethod
    def from_yaml(cls, path: str) -> "MultiViewImageToVoxelConfig":
        pass
        # with open(path, "r") as f:
        #    config = yaml.safe_load(f)
        # return cls(**config)


class VisionTransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.decoder = Transformer(hidden_size, num_layers, hidden_size // 128, 128)

    def forward(self, input: Tensor) -> Tensor:
        embeds = self.in_conv(input)
        embeds = self.decoder(embeds.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(embeds)
        return embeds


class TransformerPlane2Polar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_channels: int,
        patch_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth_channels = radius_channels
        self.in_conv = nn.Conv2d(in_channels, hidden_size * radius_channels, patch_size, stride=patch_size)
        self.transformer = Transformer(hidden_size, num_layers, hidden_size // 128, 128)
        # self.out_norm = SpatialRMSNorm(hidden_size)
        # self.nonlinear = nn.SiLU(True)
        self.out_conv = UnetAttention3d(hidden_size, out_channels, 1024, 512, 2, 2, 2, 128)

    def forward(self, multiview: Tensor, out_shape: Tuple[int, int, int]) -> Tensor:
        batch_size = multiview.shape[0]
        num_images = multiview.shape[1]
        multiview = self.in_conv(multiview.flatten(0, 1))
        multiview = multiview.view(
            batch_size,
            num_images,
            self.hidden_size,
            self.depth_channels,
            *multiview.shape[2:],
        )
        multiview_polar = torch.cat(multiview.unbind(1), dim=-1)
        multiview_polar = (
            self.transformer(multiview_polar.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(multiview_polar)
        )
        # multiview_polar = self.out_conv(multiview_polar)

        multiview_polar = ops.transforms.view_as_cartesian(multiview_polar, out_shape, "bilinear", align_corners=False)
        # multiview_polar = self.out_norm(multiview_polar)
        # multiview_polar = self.nonlinear(multiview_polar)
        multiview_polar = self.out_conv(multiview_polar)
        return multiview_polar


class UnetPlane2Polar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        polar_channels: int,
        radius_channels: int,
        base_channels: int,
        num_layers: int,
        hidden_size: int,
        num_attention_layers: int,
    ):
        super().__init__()
        self.polar_channels = polar_channels
        self.depth_channels = radius_channels
        self.in_conv = nn.Conv2d(in_channels, polar_channels * radius_channels, 1)
        self.unet = UnetLatentAttention3d(
            polar_channels, polar_channels, hidden_size, base_channels, 2, num_layers, num_attention_layers, 128
        )

    def forward(self, multiview: Tensor, out_shape: Tuple[int, int, int]) -> Tensor:
        batch_size = multiview.shape[0]
        num_images = multiview.shape[1]
        multiview = self.in_conv(multiview.flatten(0, 1))
        multiview = multiview.view(
            batch_size,
            num_images,
            self.polar_channels,
            self.depth_channels,
            *multiview.shape[2:],
        )
        multiview_polar = torch.cat(multiview.unbind(1), dim=-1)
        multiview_polar = self.unet(multiview_polar)

        multiview_polar = ops.transforms.view_as_cartesian(multiview_polar, out_shape, "bilinear", align_corners=False)
        return multiview_polar


class MultiViewImageToVoxelPipeline(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        image_autoencoderkl_model_id: str = "stabilityai/sdxl-vae",
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.voxel_encoder_latent_dim = 16
        self.plane2polar_depth_channels = 8
        self.voxel_autoencoderkl = AutoEncoderKL3d(
            num_classes,
            num_classes,
            self.voxel_encoder_latent_dim,
            64,
            2,
            3,
        )
        self.image_augmentation = ImageAugmentation()
        self.image_autoencoderkl = AutoencoderKL.from_pretrained(
            image_autoencoderkl_model_id, torch_dtype=torch.bfloat16, torchscript=True, device_map="auto"
        )
        # self.image_encoder = VisionTransformerFeatureExtractor(
        #    self.image_autoencoderkl.config.latent_channels, 8, 512, 8
        # )
        self.plane2polar = TransformerPlane2Polar(
            4, self.voxel_encoder_latent_dim * 2, self.plane2polar_depth_channels, 8, 1024, 16
        )
        # self.decoder = UnetConditionalAttention3d(
        #    512,
        #    self.voxel_encoder_latent_dim * 2,
        #    512,
        #    512,
        #    512,
        #    2,
        #    2,
        #    4,
        #    head_size=128,
        # )

        self.voxel_autoencoderkl.requires_grad_(False)
        self.image_autoencoderkl.requires_grad_(False)
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

    def prepare_multiview(self, images, voxel_shape) -> Tensor:
        batch_size = images.shape[0]
        num_images = images.shape[1]
        # images = images[:, torch.randperm(num_images)]
        with torch.no_grad():
            multiview_sample = self.prepare_image(images.flatten(0, 1))
        # multiview_features = self.image_encoder(multiview_sample)
        # multiview_features = multiview_features.view(batch_size, num_images, *multiview_features.shape[1:])
        # multiview_features = torch.cat(multiview_features.unbind(1), dim=-1)
        multiview_latent = self.plane2polar(
            multiview_sample.view(batch_size, num_images, *multiview_sample.shape[1:]), voxel_shape
        )
        return multiview_latent  # , multiview_features

    def __call__(self, input: MultiViewImageToVoxelPipelineInput) -> MultiViewImageToVoxelPipelineOutput:
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

    def random_sample_voxel(self, model_output: Tensor, voxel: Tensor) -> Tensor:
        i = random.randint(0, 32)
        j = random.randint(0, 32)
        x = i * 8
        y = j * 8
        model_output = model_output[:, :, i : i + 32, j : j + 32]
        pred_voxel = self.voxel_autoencoderkl.decode(model_output)
        voxel = voxel[:, :, x : x + 256, y : y + 256]
        pos_weight = self.influence_radial_weight(voxel)
        loss = F.binary_cross_entropy_with_logits(pred_voxel, voxel, reduction="none", pos_weight=pos_weight)
        return loss, pos_weight

    def forward(self, input: MultiViewImageToVoxelPipelineInput) -> MultiViewImageToVoxelPipelineOutput:
        if self.training:
            with torch.no_grad():
                input.images.data = self.image_augmentation(input.images.data.float()).type_as(input.images.data)

        model_dist = self.prepare_multiview(input.images, (32, 32, 4))
        model_dist = GaussianDistribution.from_latent(model_dist, latent_scale=self.voxel_autoencoderkl.latent_scale)
        model_output = model_dist.sample()

        pred_occ = self.voxel_autoencoderkl.decode(model_output)
        with torch.no_grad():
           gt_occ_dist = self.voxel_autoencoderkl.encode(input.occupancy)
        # with torch.no_grad():
        # pred_voxel = self.decode_latent_sample(model_output, input.voxel.shape[-3:])
        # loss, pos_weight = self.random_sample_voxel(model_output, voxel)
        pos_weight = self.influence_radial_weight(input.voxel)
        loss = (
            F.binary_cross_entropy_with_logits(pred_occ, input.voxel, pos_weight=pos_weight)
             + 0.0001 * model_dist.kl_div(gt_occ_dist).mean()
        )
        return MultiViewImageToVoxelPipelineOutput(
            pred_occ,
            input.voxel,
            input.occupancy,
            input.images,
            loss,
            pos_weight,
        )

    def state_dict(self, *, prefix: str = "", keep_vars: bool = False) -> dict:
        state_dict = {
            # "image_encoder": self.image_encoder.state_dict(None, prefix, keep_vars),
            "plane2polar": self.plane2polar.state_dict(None, prefix, keep_vars),
            # "decoder": self.decoder.state_dict(None, prefix, keep_vars),
        }
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # self.image_encoder.load_state_dict(state_dict["image_encoder"], strict, assign)
        self.plane2polar.load_state_dict(state_dict["plane2polar"], strict, assign)
        # self.decoder.load_state_dict(state_dict["decoder"], strict, assign)

    def influence_radial_weight(self, voxel: Tensor) -> Tensor:
        total = voxel.numel()
        num_pos = voxel.sum()
        pos_weight = math.pow((total / num_pos) * 4 * math.pi / 3, 1 / 3)
        return torch.tensor(pos_weight, device=voxel.device, dtype=voxel.dtype)


def config_model(args):
    image_autoencoderkl_model_id = os.path.join(args.save_dir, "image_autoencoderkl")
    model = None
    if os.path.exists(image_autoencoderkl_model_id):
        model = MultiViewImageToVoxelPipeline(
            num_classes=args.num_classes, image_autoencoderkl_model_id=image_autoencoderkl_model_id
        )

    else:
        model = MultiViewImageToVoxelPipeline(num_classes=args.num_classes)
        model.image_autoencoderkl.save_pretrained(image_autoencoderkl_model_id)
    try:
        path = os.path.join(args.save_dir, f"panoramic2voxel-cls{args.num_classes}.pt")
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
        strict=False,
    )

    model.to(dtype=args.dtype, device=args.device, non_blocking=True)
    model.voxel_autoencoderkl.encoder = torch.jit.script(model.voxel_autoencoderkl.encoder)
    model.voxel_autoencoderkl.decoder = torch.jit.script(model.voxel_autoencoderkl.decoder)
    model.voxel_autoencoderkl.requires_grad_(False)
    model.image_autoencoderkl.requires_grad_(False)
    return model, MultiViewImageToVoxelPipelineInput.from_nuscenes_dataset_item


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
