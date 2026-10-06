# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Small grayscale digit classifier with an explicit blank/noise class."""
import cv2
import numpy as np
import torch
from torch import nn


def digit_tensor(crop):
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    gray=cv2.resize(gray,(32,48),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return ((gray-gray.mean())/max(float(gray.std()),12.0))[None]


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(1,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,2)),nn.Flatten(),nn.Linear(512,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,11))

    def forward(self,x):
        return self.layers(x)
