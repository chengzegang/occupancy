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
from occupancy.datasets.nuscenes_dataset import (
    NuScenesDatasetItem,
    NuScenesImage,
    NuScenesDataset,
    NuScenesMixOccupancyDataset,
    NuScenesOccupancyDataset,
)
import torch.multiprocessing as mp
import torch.distributed as dist
from occupancy.models.transformer import ConditionalTransformer, CrossAttention, RMSNorm, Transformer
from occupancy.models.unet_attention_2d import UnetAttention2d, UnetAttentionEncoder2d
from occupancy.models.unet_attention_3d import UnetAttention3d, UnetAttentionEncoder3d
from occupancy.models.unet_conditional_attention_3d import (
    UnetAttentionBottleNeck3d,
    UnetConditionalAttention3d,
    UnetConditionalAttentionBottleNeck3d,
    UnetConditionalAttentionDecoderWithoutShortcut3d,
    UnetConditionalAttentionMiddleLayer3d,
)

from occupancy import ops
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from occupancy.models.unet_2d import SpatialRMSNorm, Unet2d, UnetEncoder2d, unet_decoder2d, unet_encoder2d
from occupancy.models.unet_3d import UnetDecoder3d, UnetEncoder3d, UnetLatentAttention3d
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


class OccupancyTransformerPipelineInput:
    occupancy: Tensor

    def __init__(self, occupancy: Tensor):
        self.occupancy = occupancy

    @classmethod
    def from_nuscenes_dataset_item(
        cls,
        batch: Tensor,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "OccupancyTransformerPipelineInput":
        occupancy = batch.to(device=device, dtype=dtype, non_blocking=True)

        model_input = cls(
            occupancy=occupancy,
        )
        return model_input


class OccupancyTransformerPipelineOutput:
    prediction: Tensor
    ground_truth: Tensor
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
        loss: Optional[Tensor] = None,
        pos_weight=None,
    ):
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.loss = loss
        self.pos_weight = pos_weight

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
            i, j, k = torch.where(self.ground_truth[0].argmax(dim=0).detach().cpu() > 0)
            c = self.ground_truth[0].argmax(dim=0)[i, j, k].detach().cpu()
            c = self._CMAP[c] / 255.0

            ih, jh, kh = torch.where(self.prediction[0].argmax(dim=0).detach().cpu() > 0)
            ch = self.prediction[0].argmax(dim=0)[ih, jh, kh].detach().cpu()
            ch = self._CMAP[ch] / 255.0

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"iou: {self.iou[0, 1].item():.2%}")
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.scatter(i, j, k, c=c, s=1, marker="s", alpha=0.8)
        ax.set_title("Ground Truth")
        ax.set_box_aspect((1, 1, self.ground_truth.shape[-1] / self.ground_truth.shape[-3]))
        ax.set_xlim(0, self.ground_truth.shape[-3])
        ax.set_ylim(0, self.ground_truth.shape[-2])
        ax.set_zlim(0, self.ground_truth.shape[-1])
        ax.set_zticks([])

        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(ih, jh, kh, c=ch, s=1, marker="s", alpha=0.8)
        ax.set_title("Prediction")
        ax.set_box_aspect((1, 1, self.ground_truth.shape[-1] / self.ground_truth.shape[-3]))
        ax.set_xlim(0, self.ground_truth.shape[-3])
        ax.set_ylim(0, self.ground_truth.shape[-2])
        ax.set_zlim(0, self.ground_truth.shape[-1])
        ax.set_zticks([])

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


class OccupancyTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_size: int = 1024,
        num_layers: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_conv = nn.Conv3d(self.latent_dim, self.hidden_size, 1)
        self.transformer = Transformer(
            self.hidden_size, self.num_layers, self.hidden_size // 128, 128, max_seq_length=16 * 16 * 2
        )
        self.out_conv = nn.Conv3d(self.hidden_size, self.latent_dim, 1)

    def forward(self, occ_latent: Tensor) -> Tensor:
        occ_latent = self.in_conv(occ_latent)
        occ_latent = self.transformer(occ_latent.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(occ_latent)
        output = self.out_conv(occ_latent)
        return output


@torch.jit.script
def occ_shuffle(occ: Tensor, cube_size: int = 32, shuffle_rato: float = 0.2):
    total_cubes = (occ.shape[-3] // cube_size) * (occ.shape[-2] // cube_size) * (occ.shape[-1] // cube_size)
    total_shuffle = int(total_cubes * shuffle_rato)
    cubes = torch.zeros(occ.shape[0], occ.shape[1], total_cubes, cube_size, cube_size, cube_size)
    for i in range(occ.shape[-3] // cube_size):
        for j in range(occ.shape[-2] // cube_size):
            for k in range(occ.shape[-1] // cube_size):
                ind = (
                    i * (occ.shape[-2] // cube_size) * (occ.shape[-1] // cube_size)
                    + j * (occ.shape[-1] // cube_size)
                    + k
                )
                cubes[:, :, ind] = occ[
                    :,
                    :,
                    i * cube_size : (i + 1) * cube_size,
                    j * cube_size : (j + 1) * cube_size,
                    k * cube_size : (k + 1) * cube_size,
                ]
    ind_to_shuffle = torch.randperm(total_cubes)[:total_shuffle]
    shuffle_ind = torch.randperm(total_shuffle)
    cubes[:, :, ind_to_shuffle] = cubes[:, :, ind_to_shuffle][:, :, shuffle_ind]
    shuffled = torch.zeros_like(occ)
    for i in range(occ.shape[-3] // cube_size):
        for j in range(occ.shape[-2] // cube_size):
            for k in range(occ.shape[-1] // cube_size):
                ind = (
                    i * (occ.shape[-2] // cube_size) * (occ.shape[-1] // cube_size)
                    + j * (occ.shape[-1] // cube_size)
                    + k
                )
                shuffled[
                    :,
                    :,
                    i * cube_size : (i + 1) * cube_size,
                    j * cube_size : (j + 1) * cube_size,
                    k * cube_size : (k + 1) * cube_size,
                ] = cubes[:, :, ind]
    return shuffled


class OccupancyTransformerPipeline(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        image_autoencoderkl_model_id: str = "stabilityai/sdxl-vae",
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.voxel_encoder_latent_dim = 64
        self.plane2polar_depth_channels = 8
        self.voxel_autoencoderkl = AutoEncoderKL3d(
            num_classes,
            num_classes,
            self.voxel_encoder_latent_dim,
            64,
            2,
            4,
        )
        self.image_augmentation = ImageAugmentation()
        self.image_autoencoderkl = AutoencoderKL.from_pretrained(
            image_autoencoderkl_model_id, torch_dtype=torch.bfloat16, torchscript=True, device_map="auto"
        )
        self.image_patch_conv = nn.Conv2d(3, 768, 14, stride=14)
        self.decoder = OccupancyTransformer(
            self.voxel_encoder_latent_dim,
            1024,
            12,
        )
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

    def decode(self, occ_latent, voxel_shape) -> Tensor:
        occ_latent = self.decoder(occ_latent)
        return occ_latent

    def __call__(self, input: OccupancyTransformerPipelineInput) -> OccupancyTransformerPipelineOutput:
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

    def forward(self, input: OccupancyTransformerPipelineInput) -> OccupancyTransformerPipelineOutput:
        if self.training:
            occ_latent = None
            with torch.no_grad():
                occ = occ_shuffle(input.occupancy, 32, random.random())
                occ_dist = self.voxel_autoencoderkl.encode(occ)
                occ_latent = occ_dist.sample()
                gt_sample = self.voxel_autoencoderkl.encode(input.occupancy).sample()
        model_output = self.decode(occ_latent, (16, 16, 2))
        latent_loss = F.mse_loss(model_output, gt_sample)
        pred_occ = self.voxel_autoencoderkl.decode(model_output)
        pos_weight = self.influence_radial_weight(input.occupancy)
        loss = None
        if input.occupancy.shape[1] == 1:
            loss = F.binary_cross_entropy_with_logits(
                pred_occ,
                input.occupancy,
                pos_weight=torch.tensor(3.2, device=pred_occ.device, dtype=pred_occ.dtype),
            )
        else:
            loss = F.cross_entropy(
                pred_occ,
                input.occupancy.argmax(dim=1),
                weight=pos_weight.type_as(pred_occ),
            )
        loss = loss + 0.01 * latent_loss
        return OccupancyTransformerPipelineOutput(
            pred_occ,
            input.occupancy,
            loss,
            pos_weight,
        )

    def state_dict(self, *, prefix: str = "", keep_vars: bool = False) -> dict:
        state_dict = {
            "decoder": self.decoder.state_dict(None, prefix, keep_vars),
        }
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.decoder.load_state_dict(state_dict["decoder"], strict, assign)

    def influence_radial_weight(self, voxel: Tensor) -> Tensor:
        if voxel.shape[1] == 1:
            total = voxel.numel()
            num_pos = voxel.sum()
            pos_weight = math.pow((total / num_pos) * 4 * math.pi / 3, 1 / 3)
            return torch.tensor(pos_weight, device=voxel.device, dtype=voxel.dtype)
        else:
            label = voxel.argmax(dim=1)
            population = torch.bincount(label.flatten(), minlength=18).float()
            weight = torch.pow(torch.numel(label) / population * 4 * math.pi / 3, 1 / 3)
            # weight[0] = weight[0]
            return weight


def config_model(args):
    image_autoencoderkl_model_id = os.path.join(args.save_dir, "image_autoencoderkl")
    model = None
    if os.path.exists(image_autoencoderkl_model_id):
        model = OccupancyTransformerPipeline(
            num_classes=args.num_classes, image_autoencoderkl_model_id=image_autoencoderkl_model_id
        )

    else:
        model = OccupancyTransformerPipeline(num_classes=args.num_classes)
        model.image_autoencoderkl.save_pretrained(image_autoencoderkl_model_id)
    try:
        path = os.path.join(args.save_dir, f"{args.model}-cls{args.num_classes}.pt")
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
    return model, OccupancyTransformerPipelineInput.from_nuscenes_dataset_item


def config_dataloader(args):
    if args.num_classes == 1:
        dataset = NuScenesMixOccupancyDataset(args.data_dir)
    else:
        dataset = NuScenesOccupancyDataset(args.data_dir, binary=args.num_classes == 1)
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
