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


class SegmentAwareDigitNet(nn.Module):
    """Digit classifier with an auxiliary seven-segment training head.

    The exported forward path remains the same 11 digit/background logits used
    by the browser. ``forward_with_segments`` is used only during training.
    """

    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,2)),nn.Flatten(),nn.Linear(512,128),nn.ReLU(),nn.Dropout(.15))
        self.digit_head=nn.Linear(128,11)
        self.segment_head=nn.Linear(128,7)

    def forward(self,x):
        return self.digit_head(self.features(x))

    def forward_with_segments(self,x):
        features=self.features(x)
        return self.digit_head(features),self.segment_head(features)


class SegmentDecoderDigitNet(nn.Module):
    """Predict seven on/off segments and decode them into deployed logits."""

    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,2)),nn.Flatten(),nn.Linear(512,128),nn.ReLU(),nn.Dropout(.15))
        self.segment_head=nn.Linear(128,7)
        self.foreground_head=nn.Linear(128,1)
        self.register_buffer("patterns",torch.tensor([
            [1,1,1,1,1,1,0], [0,1,1,0,0,0,0], [1,1,0,1,1,0,1],
            [1,1,1,1,0,0,1], [0,1,1,0,0,1,1], [1,0,1,1,0,1,1],
            [1,0,1,1,1,1,1], [1,1,1,0,0,0,0], [1,1,1,1,1,1,1],
            [1,1,1,1,0,1,1],
        ],dtype=torch.float32))

    def _decode(self,segment_logits,foreground_logit):
        # Spell log-sigmoid as softplus so ONNX runtimes do not underflow by
        # lowering it to log(sigmoid(x)) for confident segment predictions.
        active=-nn.functional.softplus(-segment_logits)[:,None,:]
        inactive=-nn.functional.softplus(segment_logits)[:,None,:]
        segment_scores=(self.patterns[None,:,:]*active+(1-self.patterns[None,:,:])*inactive).mean(2)
        digit_scores=segment_scores-nn.functional.softplus(-foreground_logit)
        background_score=-nn.functional.softplus(foreground_logit)
        return torch.cat((digit_scores,background_score),dim=1)

    def forward(self,x):
        features=self.features(x)
        return self._decode(self.segment_head(features),self.foreground_head(features))

    def forward_with_segments(self,x):
        features=self.features(x)
        segments=self.segment_head(features)
        return self._decode(segments,self.foreground_head(features)),segments
