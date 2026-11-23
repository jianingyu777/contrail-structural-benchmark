"""Torchvision FCN-ResNet50 wrapper for the matched RGB benchmark.

The wrapper keeps the released safetensors state-dict prefix (``network.*``)
and returns a single logit tensor, matching the other benchmark models.
"""


def build_model(pretrained=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchvision.models import ResNet50_Weights
    from torchvision.models.segmentation import fcn_resnet50

    class FCNResNet50(nn.Module):
        def __init__(self):
            super().__init__()
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.network = fcn_resnet50(
                weights=None,
                weights_backbone=weights,
                num_classes=1,
                aux_loss=False,
            )

        def forward(self, image):
            logits = self.network(image)["out"]
            return functional.interpolate(
                logits, size=image.shape[-2:], mode="bilinear", align_corners=False
            )

    return FCNResNet50()
