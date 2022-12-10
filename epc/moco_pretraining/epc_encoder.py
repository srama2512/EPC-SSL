import numpy as np
import torch.nn as nn

from epc.utils.resnet_encoder import SimpleResNetEncoder


class EPCEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
        self.projector = nn.Linear(cfg.ENCODER.hidden_size, cfg.PROJECTOR.dim)

    def forward(self, x):
        x = self.visual_encoder(x)
        x = self.visual_fc(x)
        x = self.projector(x)
        return x
