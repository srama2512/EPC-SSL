import glob
import json
import os
import random

import torch
import torchvision
from torch.utils.data import Dataset


class EPCDataset(Dataset):
    """Dataset class for loading images from EPC videos."""

    def __init__(
        self,
        root_dir,
        transform=None,
        frames_per_video=4,
    ):
        """
        Args:
            root_dir (string): Directory with all the videoes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.info_paths = glob.glob(f"{self.root_dir}/video_info_*.json")
        self.transform = transform
        self.frames_per_video = frames_per_video

    def __len__(self):
        return len(self.info_paths)

    def read_video(self, image_paths):
        frames = []
        for path in image_paths:
            img = torchvision.io.read_image(path)  # (C, H, W)
            frames.append(img)
        frames = torch.stack(frames, dim=0).float()  # (L, C, H, W)

        return frames

    def __getitem__(self, idx):
        with open(self.info_paths[idx], "r") as fp:
            infos = json.load(fp)

        # Read data
        rgb_root = os.path.join(self.root_dir, "video_frames", infos["rgb_video_name"])
        depth_root = os.path.join(
            self.root_dir, "video_frames", infos["depth_video_name"]
        )
        # Sample a subset of the video
        rgb_paths = sorted(glob.glob(f"{rgb_root}/*.jpg"))
        depth_paths = sorted(glob.glob(f"{depth_root}/*.jpg"))
        L = len(rgb_paths)
        nf = self.frames_per_video
        delta = L // nf
        s_idx = int(random.randint(0, 1 + delta))
        f_idxs = [min(s_idx + i * delta, L - 1) for i in range(0, nf)]
        rgb_paths = [rgb_paths[i] for i in f_idxs]
        depth_paths = [depth_paths[i] for i in f_idxs]

        rgb_frames = self.read_video(rgb_paths)  # (N, C, H, W)
        depth_frames = self.read_video(depth_paths)[:, 0:1]  # (N, 1, H, W)
        frames = [rgb_frames, depth_frames]
        frames = torch.cat(frames, dim=1)

        # Transformations
        if self.transform:
            frames = self.transform(frames)

        return frames
