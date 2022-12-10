import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NoiseContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.normalization in [
            "bilinear",
            "l2",
            "none",
        ], "NoiseContrastiveLoss: normalization type {} not defined".format(
            self.cfg.normalization
        )

        if self.cfg.normalization == "bilinear":
            self.bilinear_layer = nn.Linear(
                self.cfg.feature_dims, self.cfg.feature_dims, bias=False
            )

    def forward(self, z_a, z_pos):
        """
        Args:
            z_a - (bs, F) feature vector predicted
            z_pos - (bs, F) ground-truth positive feature
        """
        logits = self.compute_logits(z_a, z_pos)  # (bs, bs)
        labels = torch.arange(
            0, logits.shape[0], device=logits.device, dtype=torch.long
        )
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick from CURL:
        https://github.com/MishaLaskin/curl/blob/master/curl_sac.py
        """
        if z_a.ndim not in [2, 3]:
            raise ValueError(
                f"NoiseContrastiveLoss: compute_logits not implemented"
                f" for {z_a.ndim} dim inputs!"
            )
        if z_a.ndim == 2:
            return self.compute_single_logits(z_a, z_pos)
        elif z_a.ndim == 3:
            return self.compute_batched_logits(z_a, z_pos)
        return None

    def compute_single_logits(self, z_a, z_pos):
        """
        Estimates logits for only one set of anchors and positives.
        This is used for the loss computation.

        Args:
            z_a - (N, F) feature vector predicted
            z_pos - (N, F) ground-truth positive feature
        """
        if self.cfg.normalization == "bilinear":
            logits = torch.matmul(z_a, self.bilinear_layer(z_pos).T)
        elif self.cfg.normalization == "l2":
            logits = (
                torch.matmul(
                    F.normalize(z_a, dim=1),
                    F.normalize(z_pos, dim=1).T,
                )
                / self.cfg.temperature
            )
        else:
            logits = torch.matmul(z_a, z_pos.T)

        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_batched_logits(self, z_a, z_pos):
        """
        Estimates logits for arbitray batches of sets of anchors and positives.
        This is used for retrieval evaluation.

        Args:
            z_a - (b, N, F) feature vector N predicted
            z_pos - (b, M, F) feature vectors for M targets
        """
        if self.cfg.normalization == "bilinear":
            M = z_pos.shape[1]
            W_z_pos = self.bilinear_layer(rearrange(z_pos, "b m f -> (b m) f"))
            W_z_pos = rearrange(W_z_pos, "(b m) f -> b m f", m=M)
            logits = torch.matmul(z_a, W_z_pos.transpose(-2, -1))  # (bs, N, M)
        elif self.cfg.normalization == "l2":
            logits = (
                torch.matmul(
                    F.normalize(z_a, dim=2),
                    F.normalize(z_pos, dim=2).transpose(-2, -1),
                )
                / self.cfg.temperature
            )  # (bs, N, M)
        else:
            logits = torch.matmul(z_a, z_pos.transpose(-2, -1))

        logits = logits - torch.max(logits, 2)[0][..., None]  # (bs, N, M)
        return logits


class SelfSupervisedLoss(nn.Module):
    def __init__(self, models, cfg):
        super().__init__()
        self.cfg = cfg
        self.nce_loss = NoiseContrastiveLoss(cfg.CONTRASTIVE.clone())
        for k, v in models.items():
            setattr(self, k, v)

    def forward(self, visual_feats, poses, masks):
        """
        Args:
            visual_feats - (bs, L, F)
            poses - (bs, L, 4)
            input_masks - (bs, L)
            label_masks - (bs, L)
            cluster_ids - (bs, L, NC)
        """
        loss, _, _ = self.compute_contrastive_loss(visual_feats, poses, masks)
        return loss

    def evaluate_forward(self, visual_feats, poses, masks):
        """
        Args:
            visual_feats - (bs, L, F)
            poses - (bs, L, 4)
            input_masks - (bs, L)
            label_masks - (bs, L)

        Outputs:
            Apart from the loss, it also outputs the predicted and ground-truth feats.
        """
        return self.compute_contrastive_loss(visual_feats, poses, masks)

    def compute_contrastive_loss(self, visual_feats, poses, masks):
        """
        Args:
            visual_feats - (bs, L, F)
            poses - (bs, L, 4)
            input_masks - (bs, L)
            label_masks - (bs, L)
        """
        input_masks = masks["input_masks"]
        label_masks = masks["label_masks"]
        bs, L = label_masks.shape
        x_all = torch.cat([visual_feats, poses], 2)  # (bs, L, G1)
        # Predict label features from input features and label poses
        x_in = self.masked_sum(
            x_all, label_masks[..., None], dim=1
        ).detach()  # (bs, G1)
        pst, pen = self.transformer.pose_indices
        assert (
            pen == x_in.shape[1]
        ), "The pose values must be at the end of the feature vectors."
        if self.cfg.ignore_query_pose:
            # Query only based on time. Time is the last element of x_in.
            # The last 4 elements of x_in are (x, y, heading, time).
            x_in[..., pst : (pen - 1)] = 0
        # Remove visual features from query
        x_in[..., :pst].fill_(0)

        z_a = self.transformer(
            x_in,
            rearrange(x_all, "b l f -> l b f"),
            input_masks,
        )  # (bs, H)

        # Compute positives as average over the masked features.
        # These features are pose-agnostic. Replace pose encodings with zeros.
        pose_dims = self.transformer.pose_encoder.out_features
        padding = torch.zeros(
            x_all.shape[0], x_all.shape[1], pose_dims, device=x_all.device
        )
        x_pos = torch.cat([x_all[..., :pst], padding], 2)  # (bs, L, G2)
        x_pos = rearrange(x_pos, "b l f -> (b l) f")
        z_pos = self.transformer.fusion_encoder(x_pos)  # (bs * L, H)
        z_pos = rearrange(z_pos, "(b l) f -> b l f", b=bs)
        z_pos = self.masked_sum(z_pos, label_masks[..., None], dim=1)  # (bs, H)
        # Estimate contrastive loss
        loss = self.nce_loss(z_a, z_pos)

        return loss, z_a.detach(), z_pos.detach()

    def masked_sum(self, x, masks, dim=1):
        x = (x * masks).sum(dim=dim) / torch.clamp(masks.sum(dim=dim), min=1.0)
        return x

    def compute_logits(self, z_a, z_pos):
        return self.nce_loss.compute_logits(z_a, z_pos)
