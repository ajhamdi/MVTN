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


class MyPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        import timm.models.layers as tlayers

        img_size = tlayers.to_2tuple(img_size)
        patch_size = tlayers.to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0],
                           img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class FullCrossViewAttention(nn.Module):
    def __init__(self, model,  patch_size=16, num_views=1, feat_dim=512, num_classes=1000):
        super().__init__()
        self.model = model
        self.model.pos_embed = nn.Parameter(torch.cat(
            (self.model.pos_embed[:, 0, :].unsqueeze(1), self.model.pos_embed[:, 1::, :].repeat((1, num_views, 1))), dim=1))
        # self.model.pos_embed.retain_grad()
        self.combine_views = Rearrange(
            'b N c (h p1) (w p2) -> b c (h p1 N) (w p2)', p1=patch_size, p2=patch_size, N=num_views)
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, mvimages):
        mvimages = self.combine_views(mvimages)
        feats = self.model(mvimages)
        return self.fc(feats), feats


class WindowCrossViewAttention(nn.Module):
    def __init__(self, model,  patch_size=16, num_views=1, num_windows=1, feat_dim=512, num_classes=1000, agr_type="max"):
        super().__init__()

        assert num_views % num_windows == 0, "the number of winsows should be devidand of number of views "
        view_per_window = int(num_views/num_windows)
        model.pos_embed = nn.Parameter(torch.cat((model.pos_embed[:, 0, :].unsqueeze(
            1), model.pos_embed[:, 1::, :].repeat((1, view_per_window, 1))), dim=1))
        self.model = MVAgregate(model, agr_type=agr_type,
                                feat_dim=feat_dim, num_classes=num_classes)
        self.combine_views = Rearrange('b (Win NV) c (h p1) (w p2) -> b Win c (h p1 NV) (w p2)',
                                       p1=patch_size, p2=patch_size, Win=num_windows, NV=view_per_window)

    def forward(self, mvimages):
        mvimages = self.combine_views(mvimages)
        pred, feats = self.model(mvimages)
        return pred, feats


class MVPartSegmentation(nn.Module):
    def __init__(self,  model, num_classes, parts_per_class, parallel_head=False):
        super().__init__()
        self.num_classes = num_classes
        self.model = model
        self.multi_shape_heads = nn.ModuleList()
        self.parallel_head = parallel_head
        if parallel_head:
            for cls in range(num_classes):
                self.multi_shape_heads.append(nn.Sequential(torch.nn.Conv2d(21, 2*max(parts_per_class), kernel_size=(1, 1), stride=(1, 1)),
                                                            nn.BatchNorm2d(
                                                                2*max(parts_per_class)),
                                                            nn.ReLU(
                                                                inplace=True),
                                                            torch.nn.Conv2d(
                    2*max(parts_per_class), max(parts_per_class)+1, kernel_size=(1, 1), stride=(1, 1))
                ))
        else:
            self.multi_shape_heads.append(nn.Sequential(torch.nn.Conv2d(21, 21, kernel_size=(1, 1), stride=(1, 1)),
                                                        nn.BatchNorm2d(21),
                                                        nn.ReLU(inplace=True),
                                                        torch.nn.Conv2d(21, max(parts_per_class)+1,
                                                                        kernel_size=(1, 1), stride=(1, 1))
                                                        ))

    def forward(self, mvimages):
        features = self.model(mvctosvc(mvimages))["out"]
        if self.parallel_head:
            logits_all_shapes = []
            for cls in range(self.num_classes):
                logits_all_shapes.append(
                    self.multi_shape_heads[cls](features)[..., None])
            return torch.cat(logits_all_shapes, dim=4)
        else:
            return self.multi_shape_heads[0](features)
