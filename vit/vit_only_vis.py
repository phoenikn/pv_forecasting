import os
import time

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit.vit import *
import matplotlib.pyplot as plt


class Attention_att(Attention):
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer_att(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention_att(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        attentions = []
        for attn, ff in self.layers:
            attention_out, attention = attn(x)
            x = attention_out + x
            x = ff(x) + x
            # attention = rearrange(attention, "c (p1 p2) (h w) -> c (h p1) (w p2)", p1=16, p2=16, h=16, w=16)
            # attention = (attention - attention.min()) / (attention.max() - attention.min())
            attentions.append(attention)
        attentions = torch.cat(attentions)
        return x, attentions


class ViT_att(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        """

        :param image_size:
        :param patch_size:
        :param num_classes:
        :param dim: Dim is the length of one patch (flattened, through a linear layer)
        :param depth:
        :param heads:
        :param mlp_dim:
        :param pool:
        :param channels:
        :param dim_head:
        :param dropout:
        :param emb_dropout:
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # E.g. From bx3x224x224 to bx196x768: b x num_patches x patch_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_att(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, img):
        x = self.to_patch_embedding(img)
        # Shape of x is b x num_patches x patch_dim
        # Patch_dim = channels * patch_height * patch_width, which is flatten a small image patch
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # Parallel cat the class token with patch
        x = torch.cat((cls_tokens, x), dim=1)
        # Add each element of position embedding with the class-tokened tensor
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, attentions = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), attentions

class VIT_VIS(nn.Module):
    def __init__(self, num_min=5):
        super().__init__()
        self.num_img = num_min * 6
        self.num_min = num_min
        self.conv_before = nn.Conv2d(3, 3, (3, 3), (1, 1))
        self.vit = ViT_att(
            image_size=256,
            patch_size=16,
            num_classes=256,
            dim=1024,
            depth=4,
            heads=4,
            mlp_dim=2048,
            channels=3,
            dropout=0.25 if os.name == "nt" else 0.1,
            emb_dropout=0.25 if os.name == "nt" else 0.1
        )
        # self.batch_norm1 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv_before(x)
        # x = self.batch_norm1(x)
        x, attentions = self.vit(x)

        return x, attentions