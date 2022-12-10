import argparse
import math
import os
import os.path as osp

import dataloader
import dataset as ssl_dataset
import default as default_cfg
import models as ssl_models
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import utils as ssl_utils
from einops import rearrange, repeat
from habitat import logger
from habitat_baselines.common.tensorboard_utils import TensorboardWriter


def train(cfg):
    logger.add_filehandler(osp.join(cfg.SAVE_DIR, "train.log"))

    cfg.defrost()
    cfg.LOSS.CONTRASTIVE.feature_dims = cfg.TRANSFORMER.dim_feedforward
    cfg.freeze()

    ssl_cfg = cfg.SELF_SUPERVISED_LEARNING

    # Load dataset
    datasets = {}
    for split in ["train", "val"]:
        datasets[split] = ssl_dataset.SSLFeaturesDataset(
            osp.join(cfg.DATA_ROOT, split),
            cfg.VIDEO_LENGTH,
            num_samples_per_video=cfg.NUM_SAMPLES_PER_VIDEO,
            masking_mode=ssl_cfg.masking_mode,
            nrandom=ssl_cfg.random_segment_length,
            past_time_segment=ssl_cfg.past_time_segment,
            future_time_segment=ssl_cfg.future_time_segment,
        )

    video_batch_size = cfg.BATCH_SIZE // cfg.NUM_SAMPLES_PER_VIDEO
    data_loaders = {
        "train": dataloader.MultiEpochsDataLoader(
            datasets["train"],
            batch_size=video_batch_size,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
        ),
        "val": dataloader.MultiEpochsDataLoader(
            datasets["val"],
            batch_size=video_batch_size,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
        ),
    }

    device = torch.device("cuda:0")

    feat_shape = datasets["train"].feat_shape
    feat_size = feat_shape[0]

    # Create transformer model
    transformer_net = ssl_models.SelfSupervisedEncoder(
        feat_size + 4,
        nhead=cfg.TRANSFORMER.nhead,
        num_encoder_layers=cfg.TRANSFORMER.num_encoder_layers,
        num_decoder_layers=cfg.TRANSFORMER.num_decoder_layers,
        dim_feedforward=cfg.TRANSFORMER.dim_feedforward,
        dropout=cfg.TRANSFORMER.dropout,
        activation=cfg.TRANSFORMER.activation,
        pose_indices=(feat_size, feat_size + 4),
    )
    transformer_net.to(device)
    transformer_net.train()

    loss_models = {"transformer": transformer_net}

    # Create loss function
    ssl_loss = ssl_utils.SelfSupervisedLoss(loss_models, cfg.LOSS)
    ssl_loss.train()
    ssl_loss.to(device)

    # If pretrained model is available, resume from there
    if cfg.PRETRAINED_PATH != "":
        print(
            f"============> Loading pre-trained checkpoint from {cfg.PRETRAINED_PATH}"
        )
        loaded_state = torch.load(cfg.PRETRAINED_PATH, map_location="cpu")
        ssl_loss.load_state_dict(loaded_state["loss_state_dict"])
        for k in loss_models.keys():
            loss_models[k] = getattr(ssl_loss, k)

    # If checkpoint is available, resume from there.
    ckpt_path = osp.join(cfg.SAVE_DIR, "checkpoints", "ckpt.latest.pth")
    if os.path.isfile(ckpt_path):
        loaded_state = torch.load(ckpt_path, map_location="cpu")
        ssl_loss.load_state_dict(loaded_state["loss_state_dict"])
        for k in loss_models.keys():
            loss_models[k] = getattr(ssl_loss, k)
        start_epoch = loaded_state["extra_states"]["epoch"] + 1
        best_val_loss = loaded_state["extra_states"].get("best_val_loss", math.inf)
        logger.info(
            f"==========> Resuming training from latest checkpoint at epoch {start_epoch}"
        )
    else:
        start_epoch = 0
        best_val_loss = math.inf

    fltr = lambda x: [param for param in x if param.requires_grad]

    optimizer = optim.Adam(
        fltr(ssl_loss.parameters()),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    os.makedirs(osp.join(cfg.SAVE_DIR, "checkpoints"), exist_ok=True)

    with TensorboardWriter(cfg.SAVE_DIR, flush_secs=cfg.FLUSH_SECS) as writer:
        for epoch in range(start_epoch, cfg.EPOCHS):
            ssl_loss.train()
            running_loss = 0.0
            running_count = 0.0

            # Train model for one epoch
            for _, data in tqdm.tqdm(
                enumerate(data_loaders["train"], 0), total=len(data_loaders["train"])
            ):
                # Get the inputs
                visual_feats, poses, masks = data
                for k, v in masks.items():
                    masks[k] = v.to(device)
                visual_feats = visual_feats.to(device)
                poses = poses.to(device)
                bs, L = visual_feats.shape[:2]
                N = masks["input_masks"].shape[1]
                visual_feats = visual_feats.view(-1, *visual_feats.shape[2:])
                visual_feats = repeat(
                    visual_feats, "(b l) f -> (b n) l f", b=bs, n=N
                )  # (bs * N, L, G)
                poses = repeat(poses, "b l f -> (b n) l f", n=N)  # (bs * N, L, 4)
                # Prepare masks
                for k in ["input_masks", "label_masks"]:
                    masks[k] = rearrange(masks[k], "b n l -> (b n) l")  # (bs * N, L)
                # Forward + backward pass
                optimizer.zero_grad()
                loss = ssl_loss(visual_feats, poses, masks)
                loss.backward()
                # Optimizer step
                nn.utils.clip_grad_norm_(ssl_loss.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()
                # Update statistics
                running_loss += loss.item() * bs * N
                running_count += bs * N

            mean_train_loss = running_loss / running_count
            writer.add_scalar("train/loss", mean_train_loss, epoch)
            logger.info(
                f"===> Epoch {epoch+1}/{cfg.EPOCHS}: "
                f"train loss: {mean_train_loss:.3f}, "
            )

            if epoch % cfg.EVAL_INTERVAL == 0:
                # Evaluate on validation set
                ssl_loss.eval()
                running_val_loss = 0.0
                running_val_count = 0.0

                for _, data in tqdm.tqdm(
                    enumerate(data_loaders["val"], 0), total=len(data_loaders["val"])
                ):
                    # Get the inputs
                    visual_feats, poses, masks = data
                    visual_feats = visual_feats.to(device)  # (bs, L, F)
                    poses = poses.to(device)  # (bs, L, 4)
                    for k, v in masks.items():
                        masks[k] = v.to(device)
                    bs, L = visual_feats.shape[:2]
                    N = masks["input_masks"].shape[1]
                    visual_feats = visual_feats.view(-1, *visual_feats.shape[2:])
                    visual_feats = repeat(
                        visual_feats, "(b l) f -> (b n) l f", b=bs, n=N
                    )  # (bs * N, L, F)
                    poses = repeat(poses, "b l f -> (b n) l f", n=N)  # (bs * N, L, 4)
                    # Prepare masks
                    for k in ["input_masks", "label_masks"]:
                        masks[k] = rearrange(masks[k], "b n l -> (b n) l")
                    # Estimate loss
                    with torch.no_grad():
                        val_loss = ssl_loss(visual_feats, poses, masks)
                    # Update statistics
                    running_val_loss += val_loss.item() * bs * N
                    running_val_count += bs * N

                mean_val_loss = running_val_loss / running_val_count

                writer.add_scalar("val/loss", mean_val_loss, epoch)

                # Save model
                save_state = {
                    "loss_state_dict": ssl_loss.state_dict(),
                    "cfg": cfg,
                    "extra_states": {
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                }
                for k, v in loss_models.items():
                    save_state[f"{k}_state_dict"] = v.state_dict()

                if best_val_loss > mean_val_loss:
                    best_val_loss = mean_val_loss
                    save_state["extra_states"]["best_val_loss"] = best_val_loss
                    torch.save(
                        save_state,
                        osp.join(cfg.SAVE_DIR, "checkpoints", "ckpt.best.pth"),
                        _use_new_zipfile_serialization=False,
                    )
                torch.save(
                    save_state,
                    osp.join(cfg.SAVE_DIR, "checkpoints", "ckpt.latest.pth"),
                    _use_new_zipfile_serialization=False,
                )

                logger.info(
                    f"================= Evaluation ==================\n"
                    f"val loss: {mean_val_loss:.3f}"
                )

            if epoch % cfg.EVAL_SAVE_INTERVAL == 0:

                # Save model
                save_state = {
                    "loss_state_dict": ssl_loss.state_dict(),
                    "cfg": cfg,
                    "extra_states": {
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                }
                for k, v in loss_models.items():
                    save_state[f"{k}_state_dict"] = v.state_dict()

                torch.save(
                    save_state,
                    osp.join(cfg.SAVE_DIR, "checkpoints", f"ckpt.{epoch}.pth"),
                    _use_new_zipfile_serialization=False,
                )


def run_exp(exp_config, opts=None):
    config = default_cfg.get_config(exp_config, opts)

    train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config", type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))
