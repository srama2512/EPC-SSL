import glob
import json
import os

import numpy as np
import random
import torch
from torch.utils.data import Dataset
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector


class SSLFeaturesDataset():
    def __init__(
        self,
        root_dir,
        video_length,
        num_samples_per_video=5,
        masking_mode="temporal",
        nrandom=50,
        past_time_segment=10,
        future_time_segment=10,
    ):
        info_paths = sorted(glob.glob(f"{root_dir}/video_info_*.json"))
        self.root_dir = root_dir
        self.info_paths = info_paths
        self.video_length = video_length
        self.nspv = num_samples_per_video
        self.masking_mode = masking_mode
        # Number of random zones to sample for EPC
        self.nrandom = nrandom
        # Number of frames to sample in the past & future for DPC
        self.past_time_segment = past_time_segment
        # If dpc masking is used, how many frames in the future must be
        # sampled?
        self.future_time_segment = future_time_segment

    def __len__(self):
        return len(self.info_paths)

    def __getitem__(self, idx):
        with open(self.info_paths[idx], "r") as fp:
            infos = json.load(fp)

        visual_feats_path = os.path.join(
            self.root_dir, infos["visual_feats_name"] + ".npz"
        )
        visual_feats = np.load(visual_feats_path)["visual_feats"]
        world_poses = infos["world_poses"]
        world_poses = torch.tensor(world_poses)  # (T, 7)
        world_poses = self.convert_pose_3d_to_2d(world_poses)
        world_poses = torch.from_numpy(world_poses)

        # Append time to world_poses
        times = torch.tensor(list(range(world_poses.shape[0])), dtype=torch.float)
        world_poses = torch.cat([world_poses, times.unsqueeze(1)], 1)

        assert world_poses.shape[1] == 4
        if self.masking_mode == "temporal":
            masks = self.generate_temporal_masks(world_poses, self.nspv, self.nrandom)
        elif self.masking_mode == "dpc":
            masks = self.generate_dpc_masks(
                world_poses,
                self.nspv,
                self.past_time_segment,
                self.future_time_segment,
            )
            # Mask out the pose information
            world_poses[:, :3] = 0

        return visual_feats, world_poses, masks

    @property
    def feat_shape(self):
        feats = self[0][0]
        return feats[0].shape

    def convert_pose_3d_to_2d(self, world_poses_3d):
        world_poses_3d = np.array(world_poses_3d)  # (L, 7)
        start_pose = world_poses_3d[0]
        world_poses_2d = []
        for pose in world_poses_3d:
            world_poses_2d.append(self.estimate_pose(start_pose, pose))
        world_poses_2d = np.stack(world_poses_2d, 0)  # (L, 3)
        return world_poses_2d

    def estimate_pose(self, pose_1, pose_2):
        """Given two poses in 3D, estimate the relative pose in topdown view.
        Input conventions:
            x is rightward, -z is forward, y is upward
            heading is right-hand ruled with thumb along Y axis.
        Ouptut conventions:
            In topdown view, agent faces upward along X. Y is rightward. Heading is
            measured from X to -Y.
        """
        xyz_1 = pose_1[:3]
        quat_1 = quaternion_from_coeff(pose_1[3:])
        xyz_2 = pose_2[:3]
        quat_2 = quaternion_from_coeff(pose_2[3:])
        delta_xyz = quaternion_rotate_vector(quat_1.inverse(), xyz_2 - xyz_1)
        delta_heading = self._quat_to_xy_heading(quat_2.inverse() * quat_1)
        delta_XYH = [-delta_xyz[2].item(), delta_xyz[0].item(), delta_heading.item()]
        return np.array(delta_XYH, dtype=np.float32)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])
        heading_vector = quaternion_rotate_vector(quat, direction_vector)
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def generate_temporal_masks(self, world_poses, num_samples, nrandom):
        """
        Divide video into equally sized chunks. Sample random chunks to
        mask out.
        """
        if type(nrandom) == type([]):
            nrandom = random.randint(nrandom[0], nrandom[1])
        assert nrandom > 0
        L = world_poses.shape[0]
        masks = []
        valid_chunks = list(range(0, L, max(nrandom, 1)))
        # Select chunks to mask
        chunk_idxs = list(range(len(valid_chunks)))
        random.shuffle(chunk_idxs)
        masked_chunks = [valid_chunks[i] for i in chunk_idxs[:num_samples]]
        unmasked_chunks = [valid_chunks[i] for i in chunk_idxs[num_samples:]]
        input_masks = torch.ones(num_samples, L)
        label_masks = torch.zeros(num_samples, L)
        for i in range(num_samples):
            idx_s = masked_chunks[i]
            idx_e = idx_s + nrandom
            input_masks[:, idx_s:idx_e] = 0
            label_masks[i, idx_s:idx_e] = 1
        masks = {"input_masks": input_masks, "label_masks": label_masks}
        return masks

    def generate_dpc_masks(self, world_poses, num_samples, N, K):
        """
        Provide N frames of the video as input, and the immediately next K frames
        as the targets.

        Inspired by "Video Representation Learning by Dense Predictive Coding":
        https://arxiv.org/abs/1909.04656
        """
        L = world_poses.shape[0]
        input_masks = []
        label_masks = []
        for _ in range(num_samples):
            input_masks_i = torch.zeros(L, dtype=torch.float)
            label_masks_i = torch.zeros(L, dtype=torch.float)
            idx_s = np.random.randint(N, L - K)
            input_masks_i[(idx_s - N) : idx_s] = 1.0
            label_masks_i[idx_s : (idx_s + K)] = 1.0
            # Update outputs
            input_masks.append(input_masks_i)
            label_masks.append(label_masks_i)
        input_masks = torch.stack(input_masks, 0)
        label_masks = torch.stack(label_masks, 0)
        masks = {"input_masks": input_masks, "label_masks": label_masks}
        return masks


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.nd = len(self.datasets)
        data_points = []
        for i, dataset in enumerate(self.datasets):
            data_points += [(i, j) for j in range(len(dataset))]
        self.data_points = data_points

    def __getitem__(self, idx):
        d, i = self.data_points[idx]
        return self.datasets[d][i]

    def __len__(self):
        return len(self.data_points)
