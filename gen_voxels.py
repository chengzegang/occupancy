from occupancy.datasets import NuScenesDataset, NuScenesDatasetItem, NuScenesPointCloud
import os
import torch
from tqdm import tqdm


@torch.inference_mode
def gen(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    dataset = NuScenesDataset(data_dir)
    for ind, item in enumerate(tqdm(dataset)):
        voxel = item.lidar_top.voxel[0, 0]
        sample_token = item.lidar_top.sample_token[0]
        i, j, k = torch.where(voxel > 0)
        c = voxel[i, j, k]
        data = torch.stack([i, j, k, c], dim=0)
        out_path = os.path.join(out_dir, sample_token + ".pt")
        torch.save(data, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    gen(args.data_dir, args.out_dir)
