from functools import cached_property
import glob
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
import os
from PIL import Image as PILImage
from torch import Tensor
from tensordict import tensorclass, MemmapTensor
from nuscenes import NuScenes
import numpy as np
from torch.utils.data._utils.collate import default_collate_fn_map
import pandas as pd
import torchvision.transforms.v2.functional as TF
import warnings
import roma
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
from occupancy import ops

warnings.filterwarnings("ignore", category=UserWarning)


@tensorclass
class NuScenesImage:
    data: Tensor
    intrinsic: Tensor
    rotation: Tensor  # w, x, y, z
    translation: Tensor
    sample_token: List[str]

    @classmethod
    def load(cls, metadata: dict) -> "NuScenesImage":
        path = metadata["filename"]
        sample_token = metadata["sample_token"]
        data = TF.pil_to_tensor(PILImage.open(path))
        data = TF.to_dtype(data, torch.float32, scale=True)
        data = TF.resize(data, size=(data.size(-2) // 2, data.size(-1) // 2), antialias=True)
        data = MemmapTensor.from_tensor(data).as_tensor()
        rotation = MemmapTensor.from_tensor(roma.quat_wxyz_to_xyzw(torch.as_tensor(metadata["rotation"]))).as_tensor()
        translation = MemmapTensor.from_tensor(torch.as_tensor(metadata["translation"])).as_tensor()
        intrinsic = MemmapTensor.from_tensor(
            torch.from_numpy(np.stack(metadata["camera_intrinsic"].tolist()))
        ).as_tensor()
        return cls(
            data.unsqueeze(0).contiguous(memory_format=torch.channels_last),
            intrinsic.unsqueeze(0),
            rotation.unsqueeze(0),
            translation.unsqueeze(0),
            [sample_token],
            batch_size=[1],
        )

    @classmethod
    def collate_fn(cls, batch, *, collate_fn_map=None):
        return cls(
            torch.cat([b.data for b in batch], dim=0),
            torch.cat([b.intrinsic for b in batch], dim=0),
            torch.cat([b.rotation for b in batch], dim=0),
            torch.cat([b.translation for b in batch], dim=0),
            [token for b in batch for token in b.sample_token],
            batch_size=[len(batch)],
        )


_camera_to_car_plane = R.from_euler("xyz", [-90, 0, 90], degrees=True)
_camera_to_car_plane = roma.rotmat_to_unitquat(torch.from_numpy(_camera_to_car_plane.as_matrix())).double()


@tensorclass
class NuScenesPointCloud:
    location: Tensor
    attribute: Tensor
    voxel: Tensor
    occupancy: Tensor
    sample_token: List[str]

    def _pad_sequence(self, sequence: Tensor, max_len: int, pad_value: float = 0) -> Tensor:
        return torch.cat(
            [sequence, torch.full_like(sequence[..., :1], pad_value).expand(-1, -1, max_len - sequence.shape[-1])],
            dim=-1,
        )

    @classmethod
    def collate_fn(cls, batch, *, collate_fn_map=None):
        max_len = max([b.location.shape[-1] for b in batch])
        location = torch.cat([b._pad_sequence(b.location, max_len) for b in batch])
        attribute = torch.cat([b._pad_sequence(b.attribute, max_len) for b in batch])
        voxel = torch.cat([b.voxel for b in batch])
        occupancy = torch.cat([b.occupancy for b in batch])
        sample_token = [b.sample_token for b in batch]
        return cls(
            location,
            attribute,
            voxel,
            occupancy,
            sample_token,
            batch_size=[len(batch)],
        )

    def towards(self, image: NuScenesImage):
        batch_size = self.batch_size
        N = self.location.shape[-1]
        points = self.location.transpose(-1, -2).flatten(0, -2)
        rotations = image.rotation.flatten(0, -2).unsqueeze(-2).expand(-1, N, -1).flatten(0, 1)
        points = roma.quat_action(roma.quat_inverse(rotations.double()), points.double())
        points = roma.quat_action(_camera_to_car_plane.expand(points.shape[0], -1).type_as(points), points)
        self.location = points.reshape((*batch_size, N, 3)).transpose(-1, -2).type_as(self.location)
        return self

    @classmethod
    def _gen_voxel(cls, location, attribute, batch_size) -> Tensor:
        locs = location.flatten(0, -3)
        attrs = attribute.flatten(0, -3)
        voxel_grid = torch.stack([cls._pointcloud_to_voxelgrid(loc, attr) for loc, attr in zip(locs, attrs)])
        voxel_grid = voxel_grid.view(*batch_size, *voxel_grid.shape[1:])
        return voxel_grid

    @classmethod
    def _pointcloud_to_voxelgrid(
        cls,
        points: Tensor,
        values: Tensor,
        x_min: int = -128,
        x_max: int = 128,
        y_min: int = -128,
        y_max: int = 128,
        z_min: int = -16,
        z_max: int = 16,
        x_size: int = 256,
        y_size: int = 256,
        z_size: int = 32,
        x_offset: int = 128,
        y_offset: int = 128,
        z_offset: int = 16,
        ignore_index: int = 0,
    ) -> Tensor:
        points = points / 0.5
        # print(points[2].min())
        rot = R.from_euler("z", 90, degrees=True).as_matrix()
        rot = torch.from_numpy(rot).to(torch.float32).to(points.device)
        points[:3] = torch.matmul(rot, points[:3].type(torch.float32)).type_as(points)
        return ops.voxelize(
            points,
            values.to(torch.bfloat16),
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            x_size,
            y_size,
            z_size,
            x_offset,
            y_offset,
            z_offset,
            ignore_index,
        ).to(torch.uint8)

    @classmethod
    def _load_lidar(cls, path: str) -> Tensor:
        return torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(-1, 5)

    @classmethod
    def _load_panoptic(cls, file):
        return torch.from_numpy(np.load(file)["data"].astype(np.uint8))

    @classmethod
    def _load_full_size_occupancy(cls, path: str) -> Tensor:
        points = torch.from_numpy(np.load(path)).t()
        voxel = torch.zeros(512, 512, 64, dtype=torch.long)
        voxel[points[2].long(), points[1].long(), points[0].long()] = points[3].type(torch.long) + 1
        voxel = voxel > 0
        voxel = voxel[None, None, ...]
        return voxel

    @classmethod
    def _load_occupancy(cls, path: str) -> Tensor:
        points = torch.from_numpy(np.load(path)).t()
        voxel = torch.zeros(512, 512, 64, dtype=torch.long)
        voxel[points[2].long(), points[1].long(), points[0].long()] = points[3].type(torch.long) + 1
        voxel = voxel > 0
        voxel = voxel[None, None, ...]
        voxel = F.interpolate(voxel.float(), size=(256, 256, 32), mode="trilinear", align_corners=False)
        voxel = voxel > 0

        points_ = points.clone().float()
        points_[0] = points_[0] - 20
        points_[1] = points_[1] - 256
        points_[2] = points_[2] - 256
        points_[0] /= 20
        points_[1] /= 256
        points_[2] /= 256
        obs_ind = ops.filter_observable(points_[[2, 1, 0]], 1)
        obs_points = points[:, obs_ind]

        obs_voxel = torch.zeros(512, 512, 64, dtype=torch.long)
        obs_voxel[obs_points[2].long(), obs_points[1].long(), obs_points[0].long()] = obs_points[3].type(torch.long) + 1
        obs_voxel = obs_voxel > 0
        obs_voxel = obs_voxel[None, None, ...]
        obs_voxel = F.interpolate(obs_voxel.float(), size=(256, 256, 32), mode="trilinear", align_corners=False)
        obs_voxel = obs_voxel > 0

        return (
            points,
            obs_voxel.contiguous(memory_format=torch.channels_last_3d),
            voxel.contiguous(memory_format=torch.channels_last_3d),
        )

    @classmethod
    def load(cls, metadata: dict) -> "NuScenesPointCloud":
        sample_token = metadata["sample_token"]
        # points = cls._load_lidar(metadata["filename"]).t()
        # panoptic = cls._load_panoptic(metadata["panoptic"]["filename"])[None, None, :] + 1
        # rotation = torch.as_tensor(metadata["rotation"])
        # rotation = roma.quat_wxyz_to_xyzw(rotation)
        # voxel = cls._pointcloud_to_voxelgrid(points[0], panoptic[0])
        # points[:3] = roma.quat_action(rotation.expand(points.shape[-1], -1), points[:3].T).T
        # points = points.unsqueeze(0)
        points, voxel, occupancy = cls._load_occupancy(metadata["occupancy"])

        location = points[:3]
        panoptic = points[[3]]
        location = MemmapTensor.from_tensor(location[None, ...]).as_tensor()
        panoptic = MemmapTensor.from_tensor(panoptic[None, ...]).as_tensor()
        voxel = MemmapTensor.from_tensor(voxel).as_tensor()
        occupancy = MemmapTensor.from_tensor(occupancy).as_tensor()
        return cls(location, panoptic, voxel, occupancy, [sample_token], batch_size=[1])


@tensorclass
class NuScenesDatasetItem:
    cam_front: NuScenesImage
    cam_front_left: NuScenesImage
    cam_front_right: NuScenesImage
    cam_back: NuScenesImage
    cam_back_left: NuScenesImage
    cam_back_right: NuScenesImage

    lidar_top: NuScenesPointCloud

    @classmethod
    def load(cls, metadata: dict) -> "NuScenesDatasetItem":
        return cls(
            cam_front=NuScenesImage.load(metadata["CAM_FRONT"]),
            cam_front_left=NuScenesImage.load(metadata["CAM_FRONT_LEFT"]),
            cam_front_right=NuScenesImage.load(metadata["CAM_FRONT_RIGHT"]),
            cam_back=NuScenesImage.load(metadata["CAM_BACK"]),
            cam_back_left=NuScenesImage.load(metadata["CAM_BACK_LEFT"]),
            cam_back_right=NuScenesImage.load(metadata["CAM_BACK_RIGHT"]),
            lidar_top=NuScenesPointCloud.load(metadata["LIDAR_TOP"]),
            batch_size=[1],
        )

    @classmethod
    def collate_fn(cls, batch, *, collate_fn_map=None):
        if collate_fn_map is None:
            collate_fn_map = default_collate_fn_map
        return cls(
            NuScenesImage.collate_fn([b.cam_front for b in batch], collate_fn_map=collate_fn_map),
            NuScenesImage.collate_fn([b.cam_front_left for b in batch], collate_fn_map=collate_fn_map),
            NuScenesImage.collate_fn([b.cam_front_right for b in batch], collate_fn_map=collate_fn_map),
            NuScenesImage.collate_fn([b.cam_back for b in batch], collate_fn_map=collate_fn_map),
            NuScenesImage.collate_fn([b.cam_back_left for b in batch], collate_fn_map=collate_fn_map),
            NuScenesImage.collate_fn([b.cam_back_right for b in batch], collate_fn_map=collate_fn_map),
            NuScenesPointCloud.collate_fn([b.lidar_top for b in batch], collate_fn_map=collate_fn_map),
            batch_size=[len(batch)],
        )

    @classmethod
    def view_pointcloud_on_image(
        cls,
        image: NuScenesImage,
        pointcloud: NuScenesPointCloud,
        batch_index: int = 0,
        figsize_scale: float = 10.0,
        cmap: str = "viridis",
    ) -> plt.Figure:
        image_data = image.data[batch_index].double()
        intrinsic = image.intrinsic[batch_index].double()
        rotation = image.rotation[batch_index].double()
        translation = image.translation[batch_index].double()
        pointcloud_data = pointcloud.location[batch_index, :3].double()
        d_image = ops.view_on(image_data, pointcloud_data, intrinsic, rotation, translation)
        assert d_image.dim() == 3
        img_h, img_w = image_data.shape[-2:]
        ratio = img_h / img_w
        figsize = (figsize_scale, figsize_scale * ratio)
        with plt.ioff():
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.imshow(TF.to_pil_image(image_data))
            i, j = torch.where((d_image[-1, :, :] > 0).logical_and(torch.isfinite(d_image[-1, :, :])))
            print(i, j)
            ax.scatter(j, i, s=figsize_scale, c=d_image[-1, i, j], cmap=cmap)
        return fig


default_collate_fn_map[MemmapTensor] = default_collate_fn_map[Tensor]
default_collate_fn_map[NuScenesImage] = NuScenesImage.collate_fn
default_collate_fn_map[NuScenesPointCloud] = NuScenesPointCloud.collate_fn
default_collate_fn_map[NuScenesDatasetItem] = NuScenesDatasetItem.collate_fn


class NuScenesOccupancyDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._paths = glob.glob(os.path.join(self.data_dir, "**", "*.npy"), recursive=True)
        self._paths = np.asarray(self._paths)

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index: int):
        return NuScenesPointCloud._load_full_size_occupancy(self._paths[index])[0]


class NuScenesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        version: str = "v1.0-trainval",
        verbose: bool = True,
        force_rebuild: bool = False,
        metadata_filename: Optional[str] = "metadata.parquet",
    ):
        self.occupancy = self.build_occupancy_path(data_dir)
        self.data_dir = data_dir
        self.meta_path = os.path.join(data_dir, metadata_filename)
        if os.path.isfile(self.meta_path) and not force_rebuild:
            self.metadata = pd.read_parquet(self.meta_path)
            return
        nusc = NuScenes(
            version=version,
            dataroot=data_dir,
            verbose=verbose,
        )
        metadata = self._build_metadata(nusc)
        metadata.to_parquet(self.meta_path)

    @cached_property
    def metadata(self):
        return pd.read_parquet(os.path.join(self.data_dir, "metadata.parquet"))

    def _build_metadata(self, nusc: NuScenes) -> pd.DataFrame:
        nusc.sample
        sample_data = {d["token"]: d for d in nusc.sample_data}
        calibrated = {d["token"]: d for d in nusc.calibrated_sensor}
        metadata = []
        for index, sa in enumerate(nusc.sample):
            meta = {}
            meta.update(nusc.sample[index])
            meta["panoptic"] = nusc.panoptic[index]
            for name, token in sa["data"].items():
                sd = sample_data[token]
                sd["filename"] = os.path.join(self.data_dir, sd["filename"])
                cali = calibrated[sd["calibrated_sensor_token"]]
                sd.update(cali)
                meta[name] = sd
                meta[name]["sample_token"] = token
            meta["LIDAR_TOP"]["panoptic"] = nusc.panoptic[index]
            meta["LIDAR_TOP"]["panoptic"]["filename"] = os.path.join(
                self.data_dir, meta["LIDAR_TOP"]["panoptic"]["filename"]
            )
            metadata.append(meta)
        return pd.DataFrame(metadata)

    def get_sample_data_path(self, token: str) -> str:
        return os.path.join(self.data_dir, self.sample_data[token]["filename"])

    def build_occupancy_path(self, root: str) -> dict:
        files = glob.glob(os.path.join(root, "nuScenes-Occupancy-v0.1", "**", "*.npy"), recursive=True)
        mapping = {os.path.basename(f).split(".")[0]: f for f in files}
        return mapping

    def get_metadata(self, index: int) -> dict:
        column_index = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "LIDAR_TOP",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "panoptic",
        ]
        metadata = self.metadata.iloc[index]

        data = metadata[column_index]
        data["LIDAR_TOP"]["occupancy"] = self.occupancy[data["LIDAR_TOP"]["sample_token"]]
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int) -> NuScenesDatasetItem:
        metadata = self.get_metadata(index)
        return NuScenesDatasetItem.load(metadata)
