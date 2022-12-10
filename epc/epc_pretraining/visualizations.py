import math
import os
import subprocess as sp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import asnumpy, rearrange, repeat
from habitat.utils.visualizations.utils import append_text_to_image


def create_image_grid(x, M=None):
    """
    Given a list of images, create a square grid with them.
    M - number of images per row.
    """
    if M is None:
        N = math.ceil(math.sqrt(len(x)))
        M = N
    else:
        N = math.ceil(len(x) / float(M))
    H, W, C = x[0].shape

    grid = np.zeros((N, M, H, W, C), dtype=x[0].dtype)

    for i, x_i in enumerate(x):
        row_i = i // M
        col_i = i % M
        grid[row_i, col_i] = x_i

    grid = rearrange(grid, "m n h w c -> (m h) (n w) c")

    return grid


def generate_examples_from_masked_inputs(x, mask, K=9, M=None):
    """
    Given an array of N images and corresponding masks, select K random
    examples from images that have 1.0 in the mask.

    Args:
        x - (N, H, W, C)
        mask - (N, )
    """
    valid_idxs = np.where(mask == 1)[0]
    replace = valid_idxs.shape[0] <= K
    idxs = sorted(np.random.choice(valid_idxs, size=(K,), replace=replace))
    return create_image_grid(x[idxs], M=M)


def draw_border(x, color=(255, 0, 0), thickness=2):
    """
    Given an image, draw borders around it.

    Args:
        x - (H, W, C)
    """

    for i in range(len(color)):
        x[:thickness, :, i] = color[i]
        x[-thickness:, :, i] = color[i]
        x[:, :thickness, i] = color[i]
        x[:, -thickness:, i] = color[i]

    return x


def padded_vertical_concatenate(x, y):
    """
    Given two images of unequal sizes, this function concatenates them vertically by
    padding the smaller image to fit the width of the larger image.
    """
    hx, wx = x.shape[:2]
    hy, wy = y.shape[:2]

    if wx > wy:
        y = np.pad(y, ((0, 0), (0, wx - wy), (0, 0)))
    elif wy > wx:
        x = np.pad(x, ((0, 0), (0, wy - wx), (0, 0)))

    return np.concatenate([x, y], axis=0)


def visualize_predictions(
    label_zone_images, vis_labels, prediction_probs, K, resize_shape=None
):
    """Visualize the model predictions through retrieval of top K samples.

    Args:
        label_zone_images - (V, N, H, W, C) array
        vis_labels - (V, N) array indexing into V * N from label_zone_images
        prediction_probs - (V, N, V * N) array
    """

    V, N = label_zone_images.shape[:2]

    label_zone_images = rearrange(label_zone_images, "v n h w c -> (v n) c h w")
    if resize_shape is not None:
        label_zone_images = F.interpolate(
            torch.from_numpy(label_zone_images).float(), size=resize_shape
        )
        label_zone_images = rearrange(
            label_zone_images.numpy().astype(np.uint8), "vn c h w -> vn h w c"
        )

    visualization_images = []
    for v in range(V):
        for n in range(N):
            # Generate label images
            label_images = np.copy(label_zone_images[vis_labels[v, n].item()])
            label_images = append_text_to_image(label_images, "Ground truth")
            label_images = draw_border(label_images, color=(0, 255, 0))
            # Generate retrievals
            idxs = np.argsort(prediction_probs[v, n], axis=-1)[::-1][:K]
            retrieved_images = np.copy(label_zone_images[idxs])  # (K, H, W, C)
            retrieved_probs = prediction_probs[v, n, idxs]  # (K, )
            new_retrieved_images = [label_images]
            for r in range(retrieved_images.shape[0]):
                retrieved_images_r = append_text_to_image(
                    np.copy(retrieved_images[r]),
                    f"Probs: {retrieved_probs[r].item():.4f}",
                )
                retrieved_images_r = draw_border(retrieved_images_r, color=(0, 0, 255))
                new_retrieved_images.append(retrieved_images_r)
            new_retrieved_images = np.stack(new_retrieved_images, 0)
            grid_M = min(len(new_retrieved_images), 10)
            output_images = create_image_grid(new_retrieved_images, M=grid_M)
            visualization_images.append(output_images)

    return visualization_images


def extract_zone_images(rgb, input_masks, K=4, resize_shape=None):
    """Extract a representative image grid for a set of zones

    Args:
        rgb - (bs, L, C, H, W) tensor
        input_masks - (bs, N, L) tensor
    """

    bs, L = rgb.shape[:2]
    N = input_masks.shape[1]

    if resize_shape is not None:
        rgb = F.interpolate(
            rearrange(rgb, "b l c h w -> (b l) c h w"), size=resize_shape
        )
        rgb = rearrange(rgb, "(b l) c h w -> b l c h w", b=bs)

    rgb = (asnumpy(rgb) * 255.0).astype(np.uint8)
    rgb = rearrange(rgb, "b l c h w -> b l h w c")
    input_masks = asnumpy(input_masks)

    # Generate the source images for each of the bs * N elements
    source_images = []
    for i in range(bs):
        for j in range(N):
            source_images.append(
                generate_examples_from_masked_inputs(rgb[i], input_masks[i, j], K=K)
            )
    source_images = np.stack(source_images, 0)  # (bs * N, H, W, C)
    source_images = rearrange(source_images, "(b n) h w c -> b n h w c", b=bs)

    return source_images


def extract_zone_features(visual_feats, zone_masks, transformer, ssl_loss):
    """Extract averaged features for each zone

    Args:
        visual_feats - (bs, L, F)
        zone_masks - (bs, N, L)

    Outputs:
        zone_feats - (bs, N, F)
    """
    bs, L = visual_feats.shape[:2]
    N = zone_masks.shape[1]
    device = visual_feats.device

    visual_feats_rep = repeat(
        visual_feats, "b l f -> (b n) l f", b=bs, n=N
    )  # (bs * N, L, F)
    pose_dims = transformer.pose_encoder.out_features
    padding = torch.zeros(*visual_feats_rep.shape[:2], pose_dims, device=device)
    x_pos = torch.cat([visual_feats_rep, padding], dim=2)
    x_pos = rearrange(x_pos, "b l f -> (b l) f")  # (bs * N * L, F)
    with torch.no_grad():
        z_pos = transformer.fusion_encoder(x_pos)
    z_pos = rearrange(z_pos, "(b n l) f -> b n l f", n=N, l=L)  # (bs, N, L, F)

    zone_feats = ssl_loss.masked_sum(
        z_pos, zone_masks.unsqueeze(3), dim=2
    )  # (bs, N, F)

    return zone_feats


def dump_image(inputs):
    image, path = inputs
    cv2.imwrite(path, image[..., ::-1])


def dump_images(images, save_root, save_keys, pool=None):
    sp.call(f"mkdir -p {save_root}", shell=True)
    if pool is not None:
        image_paths = []
        for i in range(len(images)):
            image_paths.append(os.path.join(save_root, save_keys[i] + ".png"))

        for _ in tqdm.tqdm(pool.imap_unordered(dump_image, zip(images, image_paths))):
            pass
    else:
        for i, image in tqdm.tqdm(enumerate(images)):
            save_path = os.path.join(save_root, save_keys[i] + ".png")
            # Convert from RGB to BGR before writing
            cv2.imwrite(save_path, image[..., ::-1])


def get_topk_accuracy(predictions, labels, k=1):
    """Obtain top-k classification accuracy

    Args:
        predictions - (bs, N) FloatTensor logits / probabilities
        labels - (bs, ) LongTensor
        k - topk value
    """
    topk_indices = torch.topk(predictions, k=k).indices  # (bs, k)
    topk_matches = torch.sum(topk_indices == labels.unsqueeze(1), 1)
    topk_accuracy = topk_matches.float().mean().item()

    return topk_accuracy


def get_intra_video_performance(
    video_specific_data,
    ssl_loss,
    generate_visualization_masks=False,
    measure_variance_test_performance=False,
):
    """evaluate top-k classification performance for zones within a single video.

    The input contains the following keys:
        label_zone_images - (V, N, H, W, C) numpy array
        predictions - (V, N, F) tensor
        ground_truths - (V, N, F) tensor

    where
        * V is the number of videos
        * N is the number of sampled labels per video
    """
    V, N = video_specific_data["predictions"].shape[:2]

    if generate_visualization_masks:
        vis_prediction_probs = np.zeros((V, N, V, N), dtype=np.float32)
        vis_ground_truth_probs = np.zeros((V, N, V, N), dtype=np.float32)

    bs = 256
    all_logits = []
    for v in range(0, V, bs):
        # Get logits
        z_a = video_specific_data["predictions"][v : (v + bs)]  # (bs, N, F)
        z_all = video_specific_data["ground_truths"][v : (v + bs)]  # (bs, N, F)
        logits = ssl_loss.compute_logits(z_a, z_all)  # (bs, N, N)
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, 0)  # (V, N, N)
    all_unnorm_probs = torch.exp(all_logits)  # (V, N, N)
    all_probs = all_unnorm_probs / all_unnorm_probs.sum(2).unsqueeze(2)  # (V, N, N)
    all_label_idxs = repeat(torch.arange(0, N, 1, dtype=torch.long), "n -> v n", v=V)
    all_gt_probs = torch.zeros_like(all_probs)  # (V, N, N)
    all_gt_probs.scatter_(2, all_label_idxs.unsqueeze(2), 1)

    all_logits = rearrange(all_logits, "v n1 n2 -> (v n1) n2")
    all_label_idxs = rearrange(all_label_idxs, "v n -> (v n)")
    metrics = {}
    for k in [1, 2, 5]:
        if k > all_logits.shape[1]:
            continue
        topk_acc = get_topk_accuracy(all_logits, all_label_idxs, k=k)
        metrics[f"top_{k}_accuracy"] = topk_acc

    num_negatives = all_logits.shape[1] - 1

    outputs = {
        "metrics": metrics,
        "num_negatives": num_negatives,
    }

    # Generate visualization masks
    if generate_visualization_masks:
        for v in range(0, V):
            vis_prediction_probs[v, :, v, :] = asnumpy(all_probs[v])
            vis_ground_truth_probs[v, :, v, :] = asnumpy(all_gt_probs[v])

        vis_ground_truth_probs = vis_ground_truth_probs.reshape(V, N, -1)
        vis_ground_truth_labs = np.argmax(vis_ground_truth_probs, axis=2)
        outputs["vis_prediction_probs"] = rearrange(
            vis_prediction_probs, "v1 n1 v2 n2 -> v1 n1 (v2 n2)"
        )
        outputs["vis_labs"] = vis_ground_truth_labs

    return outputs


def get_inter_video_performance(
    video_specific_data,
    ssl_loss,
    generate_visualization_masks=False,
):
    """Evaluate top-k classification performance for a zone when compared to zones from
    other videos.

    The input contains the following keys:
        label_zone_images - (V, N, H, W, C) numpy array
        predictions - (V, N, F) tensor
        ground_truths - (V, N, F) tensor

    where
        * V is the number of videos
        * N is the number of sampled labels per video
    """
    V, N = video_specific_data["predictions"].shape[:2]

    prediction_logits = torch.zeros((V, N, V, N))
    ground_truth_probs = torch.zeros((V, N, V, N))

    ground_truths = rearrange(video_specific_data["ground_truths"], "v n f -> (v n) f")

    for v in range(0, V):
        # Get logits
        z_a = video_specific_data["predictions"][v]  # (N, F)
        # Estimate logits
        logits = ssl_loss.compute_logits(z_a, ground_truths)  # (N, V*N)
        logits = rearrange(logits, "b (v n) -> b v n", v=V)
        prediction_logits[v] = logits
        # Set logits to infinity for negative zones from same video
        for n in range(0, N):
            prediction_logits[v, n, v, :n] = -float("Inf")
            prediction_logits[v, n, v, (n + 1) :] = -float("Inf")
        # Estimate labels
        ground_truth_probs[v, :, v, :] = torch.eye(N)

    all_logits = rearrange(prediction_logits, "v1 n1 v2 n2 -> (v1 n1) (v2 n2)")
    gt_probs = rearrange(ground_truth_probs, "v1 n1 v2 n2 -> (v1 n1) (v2 n2)")
    _, all_labs = torch.max(gt_probs, dim=1)  # (V * N)
    metrics = {}
    for k in [1, 5, 15, 50]:
        topk_acc = get_topk_accuracy(all_logits, all_labs, k=k)
        metrics[f"top_{k}_accuracy"] = topk_acc

    num_negatives = all_logits.shape[1] - 1

    outputs = {
        "metrics": metrics,
        "num_negatives": num_negatives,
    }
    # Generate visualization masks
    if generate_visualization_masks:
        all_logits = rearrange(all_logits, "(v1 n1) f -> v1 n1 f", v1=V)
        all_unnorm_probs = torch.exp(all_logits)
        all_probs = all_unnorm_probs / all_unnorm_probs.sum(
            2, keepdim=True
        )  # (V, N, V * N)
        outputs["vis_prediction_probs"] = asnumpy(all_probs)
        outputs["vis_labs"] = asnumpy(rearrange(all_labs, "(v n) -> v n", v=V))

    return outputs


def cache_attn_weights(self, inp, output):
    """
    attn_weights - (N, L, S) where
        N is batch size
        L is target sequence length
        S is source sequence length
    """
    attn_weights = output[1]
    self.cache_attn_weights = attn_weights


def register_transformer_hooks(transformer_net):
    for layer in transformer_net.encoder.layers:
        layer.self_attn.register_forward_hook(cache_attn_weights)
    for layer in transformer_net.decoder.layers:
        layer.multihead_attn.register_forward_hook(cache_attn_weights)


def get_transformer_attention_weights(transformer_net):
    attention_weights = {}
    attention_weights["encoder"] = transformer_net.encoder.layers[
        -1
    ].self_attn.cache_attn_weights
    attention_weights["decoder"] = transformer_net.decoder.layers[
        -1
    ].multihead_attn.cache_attn_weights
    return attention_weights
