#####################################################################################################################
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from functools import partial
from torch import nn, einsum
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.initialization import initialize_decoder
from segmentation_models_pytorch.base import modules as md
from transformers import SegformerForSemanticSegmentation
#####################################################################################################################
import os, gc, time, argparse, json, copy
from math import sqrt
import math, random, cv2, librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
#####################################################################################################################
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import warnings
from functools import partial
from typing import Optional, Tuple, Type