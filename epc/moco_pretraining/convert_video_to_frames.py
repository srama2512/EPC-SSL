import argparse
import glob

import multiprocessing as mp
import os

import imageio
import tqdm


def extract_images_for_video(inputs):
    src_path, tgt_path = inputs
    os.makedirs(tgt_path, exist_ok=True)
    with imageio.get_reader(src_path) as reader:
        # print(f"Processing video {src_path}")
        for i, frame in enumerate(reader):
            imageio.imwrite(os.path.join(tgt_path, f"{i+1:04d}.jpg"), frame)


def extract_images(args):

    all_inputs = []
    for split in ["train", "val"]:
        src_paths = sorted(
            glob.glob(os.path.join(args.walkthroughs_dir, f"{split}/*.mp4"))
        )
        tgt_paths = [
            os.path.join(
                args.walkthroughs_dir,
                split,
                "video_frames",
                path.split("/")[-1][: -len(".mp4")],
            )
            for path in src_paths
        ]
        for src_path, tgt_path in zip(src_paths, tgt_paths):
            all_inputs.append((src_path, tgt_path))

    with tqdm.tqdm(total=len(all_inputs)) as pbar, mp.Pool(
        32, maxtasksperchild=1
    ) as pool:
        for _ in pool.imap_unordered(extract_images_for_video, all_inputs):
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkthroughs-dir", type=str, required=True)

    args = parser.parse_args()

    extract_images(args)
