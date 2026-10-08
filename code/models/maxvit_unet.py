"""MaxViT-B encoder and U-Net decoder used by the matched RGB benchmark."""

MAXVIT_MODEL_ID = "maxvit_base_tf_512.in21k_ft_in1k"


def build_model(pretrained=True):
    import timm
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.base import modules as md
    from segmentation_models_pytorch.base.initialization import initialize_decoder

    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.conv1 = md.Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_norm="batchnorm")
            self.conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_norm="batchnorm")

        def forward(self, x, skip=None):
            x = functional.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = torch.cat((x, skip), dim=1)
            return self.conv2(self.conv1(x))

    class MaxViTUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(MAXVIT_MODEL_ID, features_only=True, pretrained=pretrained, out_indices=(0, 1, 2, 3))
            channels = list(self.encoder.feature_info.channels())
            reductions = list(self.encoder.feature_info.reduction())
            if reductions != [2, 4, 8, 16]:
                raise ValueError(f"Unexpected MaxViT reductions: {reductions}")
            decoder_channels = (384, 192, 96, 64)
            reversed_channels = channels[::-1]
            inputs = [reversed_channels[0], *decoder_channels[:-1]]
            skips = [*reversed_channels[1:], 0]
            self.decoder = nn.ModuleList([DecoderBlock(i, s, o) for i, s, o in zip(inputs, skips, decoder_channels)])
            self.segmentation_head = smp.base.SegmentationHead(in_channels=64, out_channels=1, activation=None, kernel_size=3)
            initialize_decoder(self.decoder)

        def forward(self, x):
            features = list(self.encoder(x))[::-1]
            x = features[0]
            for index, block in enumerate(self.decoder):
                x = block(x, features[index + 1] if index + 1 < len(features) else None)
            return self.segmentation_head(x)

    return MaxViTUNet()
