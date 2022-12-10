import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar


class SimpleResNetEncoder(nn.Module):
    def __init__(
        self,
        input_shape=(256, 256, 4),
        baseplanes=32,
        ngroups=32,
        backbone_type="resnet18",
        normalize_visual_inputs=False,
    ):
        super().__init__()

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_shape[2])
        else:
            self.running_mean_and_var = nn.Sequential()

        spatial_size = input_shape[0] // 2
        input_channels = input_shape[2]
        make_backbone = getattr(resnet, backbone_type)
        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
        after_compression_flat_size = 2048
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial**2))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial,
            final_spatial,
        )

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        """
        x - (bs, C, H, W) input tensor
        """
        x = F.avg_pool2d(x, 2)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x
