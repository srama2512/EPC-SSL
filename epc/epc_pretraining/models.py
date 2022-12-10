import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from epc.utils.resnet_encoder import SimpleResNetEncoder
from epc.utils.smt_state_encoder import SMTStateEncoder


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("relu"),
)


class VisualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._feat_size = None
        self._feat_shape = None
        self.encoder_type = self.cfg.encoder_type

        self.visual_encoder = SimpleResNetEncoder(
            cfg.ENCODER.input_shape,
            baseplanes=cfg.ENCODER.baseplanes,
            ngroups=cfg.ENCODER.baseplanes // 2,
            backbone_type=cfg.ENCODER.backbone,
            normalize_visual_inputs=cfg.ENCODER.normalize_visual_inputs,
        )
        fc_input_size = np.prod(self.visual_encoder.output_shape)
        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, cfg.ENCODER.hidden_size),
            nn.ReLU(True),
        )
        self._feat_size = cfg.ENCODER.hidden_size
        self._feat_shape = (self._feat_size,)
        if cfg.ENCODER.pretrained_path != "":
            encoder_state_dict = torch.load(
                cfg.ENCODER.pretrained_path, map_location="cpu"
            )["state_dict"]
            keyword = "module.encoder_q.visual_encoder."
            visual_encoder_states = {
                k[len(keyword) :]: v
                for k, v in encoder_state_dict.items()
                if k.startswith(keyword)
            }
            self.visual_encoder.load_state_dict(visual_encoder_states)
            keyword = "module.encoder_q.visual_fc."
            visual_fc_states = {
                k[len(keyword) :]: v
                for k, v in encoder_state_dict.items()
                if k.startswith(keyword)
            }
            self.visual_fc.load_state_dict(visual_fc_states)

    def forward(self, x):
        x_cat = [x["rgb"], x["depth"]]
        x_cat = torch.cat(x_cat, dim=1)
        feats = self.visual_encoder(x_cat)
        feats = self.visual_fc(feats)
        return feats

    @property
    def feat_size(self):
        return self._feat_size

    @property
    def feat_shape(self):
        return self._feat_shape


class SelfSupervisedEncoder(SMTStateEncoder):
    def forward(self, x, memory, *args, **kwargs):
        """
        Single input case:
            Inputs:
                x - (N, input_size)
                memory - (M, N, input_size)
                memory_masks - (N, M)

        Multiple input case:
            Inputs:
                x - (K, N, input_size)
                memory - (M, N, input_size)
                memory_masks - (N, M)
            Each of the K inputs per batch element are treated independetly by generating
            an identity target mask.
        """
        assert x.dim() == 2 or x.dim() == 3
        output = None
        if x.dim() == 2:
            assert x.size(0) == memory.size(1)
            output = self.single_forward(x, memory, *args, **kwargs)
        elif x.dim() == 3:
            output = self.multi_forward(x, memory, *args, **kwargs)
        return output

    def single_forward(self, x, memory, memory_masks, goal=None, **kwargs):
        r"""Forward for a non-sequence input

        Args:
            x: (N, input_size) Tensor
            memory: The memory of encoded observations in the episode. It is a
                (M, N, input_size) Tensor.
            memory_masks: The masks indicating the set of valid memory locations
                for the current observations. It is a (N, M) Tensor.
            goal: (N, goal_dims) Tensor (optional)
        """
        # If memory_masks is all zeros for a data point, x_att will be NaN.
        # In these cases, just set memory_mask to ones and replace x_att with x.
        all_zeros_mask = (memory_masks.sum(dim=1) == 0).float().unsqueeze(1)
        memory_masks = 1.0 * all_zeros_mask + memory_masks * (1 - all_zeros_mask)

        # Compute relative pose encoding if applicable
        if self._use_pose_encoding:
            pi, pj = self._pose_indices
            x_pose = x[..., pi:]
            memory_poses = memory[..., pi:]
            x_pose_enc, memory_poses_enc = self._encode_pose(x_pose, memory_poses)
            # Update memory and observation encodings with the relative encoded poses
            x = torch.cat([x[..., :pi], x_pose_enc], dim=-1)
            memory = torch.cat([memory[..., :pi], memory_poses_enc], dim=-1)

        if self._use_goal_encoding:
            x = self.goal_observation_mlp(torch.cat([x, goal], 1))

        # Compress features
        x = self.fusion_encoder(x)
        M, bs = memory.shape[:2]
        memory = self.fusion_encoder(memory.view(M * bs, -1)).view(M, bs, -1)
        # Transformer operations
        t_masks = self._convert_masks_to_transformer_format(memory_masks)
        x_enc = self.transformer.encoder(memory, src_key_padding_mask=t_masks, **kwargs)
        x_att = self.transformer.decoder(
            x.unsqueeze(0), x_enc, memory_key_padding_mask=t_masks, **kwargs
        ).squeeze(0)
        # Mask out elements with no attention.
        x_att = x * all_zeros_mask + (1 - all_zeros_mask) * x_att

        if self._use_goal_encoding:
            x_att = torch.cat([x_att, goal], 1)

        return x_att

    def multi_forward(self, x, memory, memory_masks, **kwargs):
        r"""Forward for multiple inputs per batch

        Args:
            x: (K, N, input_size) Tensor where K is the number of inputs per batch
            memory: The memory of encoded observations in the episode. It is a
                (M, N, input_size) Tensor.
            memory_masks: The masks indicating the set of valid memory locations
                for the current observations. It is a (N, M) Tensor.

        Note: This assumes that each input per batch element will attend to the same
        set of input features.
        """
        # If memory_masks is all zeros for a data point, x_att will be NaN.
        # In these cases, just set memory_mask to ones and replace x_att with x.
        all_zeros_mask = (memory_masks.sum(dim=1) == 0).float().unsqueeze(1)
        memory_masks = 1.0 * all_zeros_mask + memory_masks * (1 - all_zeros_mask)

        # Compute relative pose encoding if applicable
        if self._use_pose_encoding:
            pi, pj = self._pose_indices
            # All inputs of a batch element are assumed to be reasonably close to
            # each other. So, the average pose is computed across inputs in a batch element.
            # The relative encoding is performed w.r.t that average pose.
            x_pose = x[..., pi:]  # (K, N, 4)
            x_pose_avg = x_pose.mean(dim=0)  # (N, 4)
            memory_poses = memory[..., pi:]  # (M, N, 4)
            _, x_pose_enc = self._encode_pose(x_pose_avg, x_pose)  # (K, N, G)
            _, memory_poses_enc = self._encode_pose(
                x_pose_avg, memory_poses
            )  # (M, N, G)

            # Update memory and observation encodings with the relative encoded poses
            x = torch.cat([x[..., :pi], x_pose_enc], dim=-1)  # (K, N, F)
            memory = torch.cat(
                [memory[..., :pi], memory_poses_enc], dim=-1
            )  # (K, N, F)

        # Compress features
        K, bs = x.shape[:2]
        M = memory.shape[0]
        x = self.fusion_encoder(x.view(K * bs, -1)).view(K, bs, -1)
        memory = self.fusion_encoder(memory.view(M * bs, -1)).view(M, bs, -1)

        # Transformer operations
        t_masks = self._convert_masks_to_transformer_format(memory_masks)
        ## Identity mask to prevent attention b/w the decoder inputs
        d_masks = self._convert_masks_to_transformer_format(
            torch.eye(K, K, device=x.device)
        )
        x_enc = self.transformer.encoder(memory, src_key_padding_mask=t_masks, **kwargs)
        x_att = self.transformer.decoder(
            x, x_enc, memory_key_padding_mask=t_masks, tgt_mask=d_masks, **kwargs
        )  # (K, N, F)

        # Mask out elements with no attention.
        all_zeros_mask = all_zeros_mask.unsqueeze(0)  # (1, N, 1)
        x_att = x * all_zeros_mask + (1 - all_zeros_mask) * x_att

        return x_att
