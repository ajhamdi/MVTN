# to import files from parent dir
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import mvctosvc
from util import batch_tensor, unbatch_tensor
import torch
import numpy as np
from torch import nn
from torch._six import inf
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class ViewMaxAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.max(unbatch_tensor(self.model(batch_tensor(
            mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True), dim=1)[0]
        return pooled_view.squeeze()


class ViewAvgAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.mean(unbatch_tensor(self.model(batch_tensor(
            mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True), dim=1)
        return pooled_view.squeeze()


class UnrolledDictModel(nn.Module):
    "a helper class that unroll pytorch models that return dictionaries instead of tensors"

    def __init__(self,  model, keyword="out"):
        super().__init__()
        self.model = model
        self.keyword = keyword

    def forward(self, x):
        return self.model(x)[self.keyword]


class MVAgregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000):
        super().__init__()
        self.agr_type = agr_type
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAgregate(model=model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAgregate(model=model)

    def forward(self, mvimages):
        pooled_view = self.aggregation_model(mvimages)
        predictions = self.fc(pooled_view)
        return predictions, pooled_view

