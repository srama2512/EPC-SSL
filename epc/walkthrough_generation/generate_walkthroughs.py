import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
from typing import List, Optional

import imageio
import numpy as np
import torch
import tqdm

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

import habitat_sim
from habitat.utils.geometry_utils import quaternion_from_coeff

MAX_DEPTH = 10.0


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    for im in images:
        writer.append_data(im)
    writer.close()


def make_configuration(scene_path, gpu_device_id=0):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.gpu_device_id = gpu_device_id

    # sensor configurations
    camera_resolution = [256, 342]
    sensor_specs = []

    rgba_camera_spec = habitat_sim.SensorSpec()
    rgba_camera_spec.uuid = "rgba"
    rgba_camera_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_spec.resolution = camera_resolution
    rgba_camera_spec.position = [0.0, 0.88, 0.0]
    rgba_camera_spec.parameters["hfov"] = "79"
    sensor_specs.append(rgba_camera_spec)

    depth_camera_spec = habitat_sim.SensorSpec()
    depth_camera_spec.uuid = "depth"
    depth_camera_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_camera_spec.resolution = camera_resolution
    depth_camera_spec.position = [0.0, 0.88, 0.0]
    depth_camera_spec.parameters["hfov"] = "79"
    sensor_specs.append(depth_camera_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.radius = 0.18
    agent_cfg.height = 0.88
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def proc_obs(obs):
    rgb_obs = obs["rgba"][..., :3]  # (H, W, 3)
    depth_obs = obs["depth"]  # (H,  W)
    depth_obs = (np.clip(depth_obs / MAX_DEPTH, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb_obs, depth_obs


def generate_walkthrough_for_video(inputs):
    info_path, device_id = inputs
    with open(info_path, "r") as fp:
        infos = json.load(fp)
    scene_path = infos["scene_id"]
    # Load simulator
    cfg = make_configuration(scene_path, gpu_device_id=device_id)
    sim = habitat_sim.Simulator(cfg)
    # Generate observations for RGB and depth
    rgb_frames = []
    depth_frames = []
    for world_pose in infos["world_poses"]:
        position = np.array(world_pose[:3])
        rotation = quaternion_from_coeff(world_pose[3:])
        # Set agent state
        agent = sim.get_agent(0)
        new_state = sim.get_agent(0).get_state()
        new_state.position = position
        new_state.rotation = rotation
        new_state.sensor_states = {}
        agent.set_state(new_state)
        # Get observations
        obs = sim.get_sensor_observations()
        rgb_obs, depth_obs = proc_obs(obs)
        rgb_frames.append(rgb_obs)
        depth_frames.append(depth_obs)
    # Save videos
    save_dir = os.path.dirname(info_path)
    rgb_video_name = infos["rgb_video_name"]
    images_to_video(rgb_frames, save_dir, rgb_video_name, fps=10, quality=10)
    depth_video_name = infos["depth_video_name"]
    images_to_video(depth_frames, save_dir, depth_video_name, fps=10, quality=10)
    # Close simulator
    sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infos-root", type=str, default="./data/walkthroughs/strong_exploration/"
    )
    parser.add_argument("--num-workers-per-gpu", type=int, default=12)

    args = parser.parse_args()

    info_paths = []
    for split in ["train", "val"]:
        info_paths += sorted(
            glob.glob(os.path.join(args.infos_root, split, "video_info_*.json"))
        )

    context = mp.get_context("forkserver")
    n_devices = torch.cuda.device_count()
    inputs = []
    for i, path in enumerate(info_paths):
        inputs.append((path, i % n_devices))

    with context.Pool(
        args.num_workers_per_gpu * n_devices, maxtasksperchild=1
    ) as pool, tqdm.tqdm(total=len(inputs)) as pbar:
        for _ in pool.imap_unordered(generate_walkthrough_for_video, inputs):
            pbar.update()
