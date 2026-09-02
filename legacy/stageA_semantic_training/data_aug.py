# from __init__ import *

# # 256 * 256
# transform_train = albu.Compose([
#     albu.ShiftScaleRotate(
#         scale_limit=0.2,
#         rotate_limit=180,
#         shift_limit=0.3,
#         border_mode=0,
#         p=1,
#     ),
#     albu.PadIfNeeded(
#         min_height=256, min_width=256, always_apply=True, border_mode=0
#         ),
#     albu.Resize(256, 256),
#     albu.OneOf(
#         [
#             albu.RandomBrightnessContrast(
#             brightness_limit=0.2, contrast_limit=0.3, p=1
#             ),
#             albu.RandomGamma(gamma_limit=(20, 100), p=1),
#             ],
#             p=0.5,
#             ),
#     ToTensorV2()
#     ])

# transform_val = albu.Compose([
#     albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
#     albu.Resize(256, 256),
#     ToTensorV2()
#     ])

# transform_test = albu.Compose([
#     albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
#     albu.Resize(256, 256),
#     ToTensorV2()
#     ])

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 训练增强：注意所有几何变换都要给 mask_value=0，保证 mask 不被“糊”
transform_train = A.Compose([
    A.ShiftScaleRotate(
        scale_limit=0.1,
        rotate_limit=180,
        shift_limit=0.2,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,            # image 边缘填充值
        mask_value=0,       # mask 边缘填充值
        p=1.0
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.PadIfNeeded(min_height=256, min_width=256, always_apply=True,
                  border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),  # image 线性；mask 会自动最近邻
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(20, 100), p=1.0),
    ], p=0.5),
    ToTensorV2()
])

# 验证/测试：只做几何对齐
transform_val = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, always_apply=True,
                  border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
    ToTensorV2()
])

transform_test = transform_val

