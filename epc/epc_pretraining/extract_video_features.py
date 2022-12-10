import argparse
import json
import multiprocessing as mp
import os
import os.path as osp

import default as default_cfg
import models as ssl_models
import numpy as np
import torch
import tqdm
from einops import asnumpy

from epc.utils.video_dataset import CenterCrop, VideoDataset


def extract_scene_episode_id_from_info_path(path):
    """
    path is of the form {data_root}/video_info_{scene_id}_{episode_id}.json
    """
    _, _, scene_id, episode_id = path.split("/")[-1][: -len(".json")].split("_")
    return scene_id, episode_id


def _worker_extract_video_features(inputs):
    cfg, split, video_indices, save_root, device = inputs

    transform = CenterCrop((cfg.IMAGE_HEIGHT, cfg.IMAGE_HEIGHT))

    data_root = cfg.DATA_ROOT
    dataset = VideoDataset(
        osp.join(data_root, split),
        cfg.IMAGE_WIDTH,
        cfg.IMAGE_HEIGHT,
        cfg.VIDEO_LENGTH,
        transform=transform,
    )

    # Create encoder model
    visual_encoder = ssl_models.VisualEncoder(cfg.VISUAL_ENCODER.RESNET)
    visual_encoder.to(device)
    visual_encoder.eval()  # Keep visual encoder frozen

    for video_index in video_indices:
        # Compute save path
        scene_id, episode_id = extract_scene_episode_id_from_info_path(
            dataset.info_paths[video_index]
        )
        feat_save_path = osp.join(
            save_root, f"video_features_{scene_id}_{episode_id}.npy"
        )
        # Skip if already present
        if os.path.isfile(feat_save_path):
            continue
        info_save_path = osp.join(save_root, f"video_info_{scene_id}_{episode_id}.json")
        # Compute json infos
        with open(dataset.info_paths[video_index], "r") as fp:
            orig_infos = json.load(fp)
        infos = {
            "world_poses": orig_infos["world_poses"],
            "episode_id": orig_infos["episode_id"],
            "scene_id": orig_infos["scene_id"],
            "visual_feats_name": f"video_features_{scene_id}_{episode_id}.npy",
        }
        # Extract video features
        frames = dataset[video_index]
        extraction_step = 50
        visual_feats = []
        for j in range(0, frames["rgb"].shape[0], extraction_step):
            frames_j = {
                k: v[j : (j + extraction_step)].to(device) for k, v in frames.items()
            }
            with torch.no_grad():
                visual_feats_j = visual_encoder(frames_j).detach()
            visual_feats.append(asnumpy(visual_feats_j))  # (l, C, H, W)
        visual_feats = np.concatenate(visual_feats, axis=0)  # (L, F, 16, 16)
        # Save video features
        np.savez_compressed(feat_save_path, visual_feats=visual_feats)
        # Save infos
        with open(info_save_path, "w") as fp:
            json.dump(infos, fp)

    return len(video_indices)


def main(args):

    cfg = default_cfg.get_config()
    cfg.merge_from_file(args.exp_config)

    cfg.defrost()
    cfg.freeze()

    try:
        os.mkdir(args.save_root)
    except Exception:
        pass

    n_devices = torch.cuda.device_count()
    n_processes = args.num_processes_per_gpu * n_devices
    # Extract features for train, val splits
    for split in ["train", "val"]:
        data_root = cfg.DATA_ROOT
        dataset = VideoDataset(
            osp.join(data_root, split),
            cfg.IMAGE_WIDTH,
            cfg.IMAGE_HEIGHT,
            cfg.VIDEO_LENGTH,
        )

        # Copy configs per process
        cfg_per_process = [cfg.clone() for _ in range(n_processes)]

        # Assign splits per process
        splits_per_process = [split for _ in range(n_processes)]

        # Split videos into approximately equal sized chunks
        num_videos = len(dataset)
        indices_per_process = np.array_split(range(num_videos), n_processes)

        # Assign save_roots per process
        try:
            os.mkdir(osp.join(args.save_root, split))
        except Exception:
            pass
        save_roots_per_process = [
            osp.join(args.save_root, split) for _ in range(n_processes)
        ]

        # Assign devices to each process
        devices_per_process = []
        for i in range(n_processes):
            device_id = i % n_devices
            devices_per_process.append(torch.device(f"cuda:{device_id:d}"))

        with mp.get_context("spawn").Pool(
            n_processes, maxtasksperchild=1
        ) as p, tqdm.tqdm(total=num_videos) as pbar:
            for n_videos_done in p.imap_unordered(
                _worker_extract_video_features,
                zip(
                    cfg_per_process,
                    splits_per_process,
                    indices_per_process,
                    save_roots_per_process,
                    devices_per_process,
                ),
            ):
                pbar.update(n_videos_done)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config", type=str, required=True)
    parser.add_argument("--num-processes-per-gpu", type=int, default=4)
    parser.add_argument("--save-root", type=str, required=True)

    args = parser.parse_args()

    main(args)
