from __future__ import division, print_function

import glob
import json
import os

import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset


# Code borrowed from https://github.com/MohsenFayyaz89/PyTorch_Video_Dataset/blob/master/GeneralVideoDataset.py
class RandomCrop:
    """Randomly Crop the frames in a clip."""

    def __init__(self, output_size):
        """
        Args:
          output_size (tuple or int): Desired output size. If int, square crop
          is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        if h - new_h > 0:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if w - new_w > 0:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0

        clip = clip[:, :, top : top + new_h, left : left + new_w]

        return clip


class CenterCrop:
    """Crop the center of frames in a clip."""

    def __init__(self, output_size):
        """
        Args:
          output_size (tuple or int): Desired output size. If int, square crop
          is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        clip = clip[:, :, top : top + new_h, left : left + new_w]

        return clip


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        width,
        height,
        length,
        transform=None,
    ):
        info_paths = sorted(glob.glob(f"{root_dir}/video_info_*.json"))
        self.root_dir = root_dir
        self.info_paths = info_paths
        self.transform = transform
        self.video_shape = (length, height, width)
        self.video_length = length
        decord.bridge.set_bridge("torch")

    def __len__(self):
        return len(self.info_paths)

    def read_video(self, video_file):
        # Open the video file
        vr = VideoReader(video_file, ctx=cpu(0))
        num_frames, height, width = self.video_shape
        try:
            frames = vr.get_batch(range(num_frames))
            frames = frames.permute(0, 3, 1, 2).float()
            failed_clip = False
        except Exception:
            frames = torch.zeros(num_frames, 3, height, width)
            failed_clip = True

        return frames, failed_clip

    def __getitem__(self, idx):
        with open(self.info_paths[idx], "r") as fp:
            infos = json.load(fp)

        rgb_path = os.path.join(self.root_dir, infos["rgb_video_name"] + ".mp4")
        depth_path = os.path.join(self.root_dir, infos["depth_video_name"] + ".mp4")
        rgb_frames = self.read_video(rgb_path)[0]  # (L, C, H, W)
        depth_frames = self.read_video(depth_path)[0]  # (L, C, H, W)
        frames = {"rgb": rgb_frames, "depth": depth_frames}

        if self.transform:
            # Concatenate the frames and jointly transform them
            frame_keys = list(frames.keys())
            concat_frames = [frames[k] for k in frame_keys]
            concat_frames = torch.cat(concat_frames, 0)
            concat_frames = self.transform(concat_frames)
            transformed_frames = {}
            count = 0
            for _, k in enumerate(frame_keys):
                transformed_frames[k] = concat_frames[
                    count : (count + frames[k].shape[0])
                ]
                count += frames[k].shape[0]
            frames = transformed_frames

        for k in frames.keys():
            frames[k] /= 255.0
        frames["depth"] = frames["depth"][:, 0:1]

        return frames
