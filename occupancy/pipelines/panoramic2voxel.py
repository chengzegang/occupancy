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
from occupancy.pipelines.occ_transformer import OccupancyTransformer
from .autoencoderkl_3d import AutoEncoderKL3d, GaussianDistribution
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import InterpolationMode
from torch.utils.data import Subset
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
import warnings
import torchvision.transforms.v2.functional as TF

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
        with torch.autocast("cuda", torch.float32):
            return self.transform(x)


class MultiViewImageToVoxelPipelineInput:
    images: Tensor
    occupancy: Tensor

    def __init__(self, images: Tensor, occupancy: Tensor):
        self.images = images
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

        model_input = cls(
            images=images.data,
            occupancy=occupancy,
        )
        return model_input


class MultiViewImageToVoxelPipelineOutput:
    prediction: Tensor
    ground_truth: Tensor
    images: Tensor
    loss: Optional[Tensor] = None
    pos_weight: Optional[Tensor] = None

    _CMAP = np.asarray(
        [  # RGB.
            (np.nan, np.nan, np.nan),  # None 0
            (0, 0, 0),  # Black. noise 1
            (112, 128, 144),  # Slategrey barrier 2
            (220, 20, 60),  # Crimson bicycle 3
            (255, 127, 80),  # Orangered bus 4
            (255, 158, 0),  # Orange car 5
            (233, 150, 70),  # Darksalmon construction 6
            (255, 61, 99),  # Red motorcycle 7
            (0, 0, 230),  # Blue pedestrian 8
            (47, 79, 79),  # Darkslategrey trafficcone 9
            (255, 140, 0),  # Darkorange trailer 10
            (255, 99, 71),  # Tomato truck 11
            (0, 207, 191),  # nuTonomy green driveable_surface 12
            (175, 0, 75),  # flat other 13
            (75, 0, 75),  # sidewalk 14
            (112, 180, 60),  # terrain
            (222, 184, 135),  # Burlywood mannade
            (0, 175, 0),  # Green vegetation
        ]
    )

    def __init__(
        self,
        prediction: Tensor,
        ground_truth: Tensor,
        images: Tensor,
        loss: Optional[Tensor] = None,
        pos_weight=None,
    ):
        self.prediction = prediction
        self.ground_truth = ground_truth
        self.images = images
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


class MultiViewImageToVoxelModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 16,
        radius_channels: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = 1024
        self.radius_channels = radius_channels
        self.patch_conv = nn.Conv2d(in_channels, self.hidden_size, 4, stride=4)

        self.positional_embeds = nn.Embedding(10000, self.hidden_size)
        self.register_buffer("positional_ids", torch.arange(10000).view(1, -1).long())
        self.p_encoder = Transformer(self.hidden_size, 8, self.hidden_size // 128, 128, max_seq_length=10000)
        self.k_encoder = Transformer(self.hidden_size, 8, self.hidden_size // 128, 128, max_seq_length=10000)
        self.v_encoder = Transformer(self.hidden_size, 8, self.hidden_size // 128, 128, max_seq_length=10000)
        self.decoder = Transformer(self.hidden_size, 8, self.hidden_size // 128, 128, max_seq_length=10000)
        self.occ_norm = RMSNorm(self.hidden_size)
        self.nonlinear = nn.SiLU(True)
        self.occ_proj = nn.Linear(self.hidden_size, 64)
        self.occ_transformer = OccupancyTransformer(64, 1024, 12)

    def forward(self, multiview: Tensor, out_shape: Tuple[int, int, int]) -> Tensor:
        # multiview = torch.cat(multiview.unbind(1), dim=-1)
        seq_len = out_shape[0] * out_shape[1] * out_shape[2]
        q = self.positional_embeds(self.positional_ids[:, :seq_len])

        q = q.expand(multiview.shape[0], -1, -1)

        # kv_embeds = self.patch_conv(multiview).flatten(2).transpose(1, 2)
        pkv_embeds = self.p_encoder(multiview)
        k = self.k_encoder(pkv_embeds)
        v = self.v_encoder(pkv_embeds)

        q = q.view(q.shape[0], q.shape[1], -1, 128).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], -1, 128).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], -1, 128).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(2)
        occ_latent = self.decoder(out)

        occ_latent = self.occ_norm(occ_latent)
        occ_latent = self.nonlinear(occ_latent)
        occ_latent = self.occ_proj(occ_latent)
        occ_latent = self.occ_transformer(occ_latent.transpose(1, 2).reshape(out.shape[0], -1, *out_shape))
        return occ_latent


class MultiViewImageToVoxelPipeline(nn.Module):
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
        from torchvision import models
        from transformers import CLIPVisionModel

        self.image_feature = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

        self.decoder = MultiViewImageToVoxelModel(
            4,
            self.voxel_encoder_latent_dim,
            self.plane2polar_depth_channels,
        )
        self.voxel_autoencoderkl.requires_grad_(False)
        self.image_autoencoderkl.requires_grad_(False)

    def decode(self, images, voxel_shape) -> Tensor:
        batch_size = images.shape[0]
        num_images = images.shape[1]
        with torch.no_grad():
            images = images.flatten(0, 1)
            images = F.interpolate(images, (224, 224), mode="bicubic", antialias=True)
            images = TF.normalize(images, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
            chunks = []
            for i in range(2):
                for j in range(2):
                    ck = images[:, :, i * 224 : (i + 1) * 224, j * 224 : (j + 1) * 224]
                    chunks.append(self.image_feature(ck).last_hidden_state[:, 1:])
            chunks = torch.cat(chunks, dim=1)
            # multiview_sample = self.image_feature(images).last_hidden_state[:, 1:]
            multiview_sample = chunks.view(batch_size, num_images, *chunks.shape[1:]).flatten(1, 2)

        multiview_latent = self.decoder(multiview_sample, voxel_shape)
        return multiview_latent

    def __call__(self, input: MultiViewImageToVoxelPipelineInput) -> MultiViewImageToVoxelPipelineOutput:
        return super().__call__(input)

    def forward(self, input: MultiViewImageToVoxelPipelineInput) -> MultiViewImageToVoxelPipelineOutput:
        if self.training:
            with torch.no_grad():
                input.images.data = self.image_augmentation(input.images.data.float()).type_as(input.images.data)
                input.images.data = input.images.data[:, torch.randperm(input.images.shape[1])]
                gt_occ = self.voxel_autoencoderkl.encode(input.occupancy).sample()
        model_output = self.decode(input.images, (16, 16, 2))
        pred_occ = self.voxel_autoencoderkl.decode(model_output)
        pos_weight = self.influence_radial_weight(input.occupancy)
        latent_loss = F.mse_loss(model_output, gt_occ)
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
                ignore_index=1,
            )
        occ = input.occupancy.argmax(dim=1) > 0
        num_occ = torch.sum(occ)
        num_non_occ = torch.numel(occ) - num_occ
        ratio = num_occ / num_non_occ
        loss = 100 * ratio * (loss + latent_loss)
        return MultiViewImageToVoxelPipelineOutput(
            pred_occ,
            input.occupancy,
            input.images,
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
            return weight


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
    model.decoder.occ_transformer.load_state_dict(
        torch.load(
            os.path.join(args.save_dir, f"occ_transformer-cls{args.num_classes}.pt"),
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
    model.decoder.occ_transformer.requires_grad_(False)
    model.image_feature.requires_grad_(False)
    return model, MultiViewImageToVoxelPipelineInput.from_nuscenes_dataset_item


def config_dataloader(args):
    dataset = NuScenesDataset(args.data_dir, binary=args.num_classes == 1)
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
