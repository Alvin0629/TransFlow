# --------------------------------------------------------
# GroupAttn Blocks
# Copyright (c) 2021 Meituan
# Licensed under The Apache 2.0 License [see LICENSE for details]
# Written by Xinjie Li, Xiangxiang Chu
# --------------------------------------------------------
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Attention
from .attention import MultiHeadAttention, LinearPositionEmbeddingSine
from utils.utils import coords_grid, bilinear_sampler

Size_ = Tuple[int, int]

class GroupAttnRPEContext(nn.Module):   
    """ Latent cost tokens attend to different group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1, cfg=None, vert_c_dim=0):
        super(GroupAttnRPEContext, self).__init__()
        assert ws != 1
        assert cfg is not None
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert cfg.cost_embed_token_num % 5 == 0, "cost_embed_token_num should be divided by 5."
        assert vert_c_dim > 0, "vert_c_dim should not be 0"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.vert_c_dim = vert_c_dim

        self.cfg = cfg

        self.context_proj = nn.Linear(256, vert_c_dim)
        self.q = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        C_qk = C+self.vert_c_dim
        H, W = size
        batch_num = B // 5

        context = context.repeat(B//context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H*W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)

        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)

        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        padded_N = Hp*Wp

        coords = coords_grid(B, Hp, Wp).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk)
        coords_enc = coords_enc.reshape(B, Hp, Wp, C_qk)

        q = self.q(x_qk + coords_enc).reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        q = q.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v = self.v(x)
        k = self.k(x_qk + coords_enc)
        # concate and do shifting operation together
        kv = torch.cat([k, v], dim=-1)
        kv_up = torch.cat([kv[:batch_num, self.ws:Hp, :, :], kv[:batch_num, Hp-self.ws:Hp, :, :]], dim=1)
        kv_down = torch.cat([kv[batch_num:batch_num*2, :self.ws, :, :], kv[batch_num:batch_num*2, :Hp-self.ws, :, :]], dim=1)
        kv_left = torch.cat([kv[batch_num*2:batch_num*3, :, self.ws:Wp, :], kv[batch_num*2:batch_num*3, :, Wp-self.ws:Wp, :]], dim=2)
        kv_right = torch.cat([kv[batch_num*3:batch_num*4, :, :self.ws, :], kv[batch_num*3:batch_num*4, :, :Wp-self.ws, :]], dim=2)
        kv_center = kv[batch_num*4:batch_num*5, :, :, :]
        kv_shifted = torch.cat([kv_up, kv_down, kv_left, kv_right, kv_center], dim=0)
        k, v = torch.split(kv_shifted, [self.dim, self.dim], dim=-1)
        
        k = k.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        k = k.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v = v.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        v = v.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GroupAttnRPE(nn.Module):  
    """ Latent cost tokens attend to different group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1, cfg=None):
        super(GroupAttnRPE, self).__init__()
        assert ws != 1
        assert cfg is not None
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert cfg.cost_embed_token_num % 5 == 0, "cost_embed_token_num should be divided by 5."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.cfg = cfg

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        batch_num = B // 5 
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        padded_N = Hp*Wp

        coords = coords_grid(B, Hp, Wp).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        coords_enc = coords_enc.reshape(B, Hp, Wp, C)

        q = self.q(x + coords_enc).reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        q = q.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v = self.v(x)
        k = self.k(x + coords_enc)
        # concate and do shifting operation together
        kv = torch.cat([k, v], dim=-1)
        kv_up = torch.cat([kv[:batch_num, self.ws:Hp, :, :], kv[:batch_num, Hp-self.ws:Hp, :, :]], dim=1)
        kv_down = torch.cat([kv[batch_num:batch_num*2, :self.ws, :, :], kv[batch_num:batch_num*2, :Hp-self.ws, :, :]], dim=1)
        kv_left = torch.cat([kv[batch_num*2:batch_num*3, :, self.ws:Wp, :], kv[batch_num*2:batch_num*3, :, Wp-self.ws:Wp, :]], dim=2)
        kv_right = torch.cat([kv[batch_num*3:batch_num*4, :, :self.ws, :], kv[batch_num*3:batch_num*4, :, :Wp-self.ws, :]], dim=2)
        kv_center = kv[batch_num*4:batch_num*5, :, :, :]
        kv_shifted = torch.cat([kv_up, kv_down, kv_left, kv_right, kv_center], dim=0)
        k, v = torch.split(kv_shifted, [self.dim, self.dim], dim=-1)
        
        k = k.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        k = k.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v = v.reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads).transpose(2, 3)
        v = v.reshape(B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LocallyGroupedAttnRPEContext(nn.Module):  
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1, vert_c_dim=0):
        assert ws != 1
        super(LocallyGroupedAttnRPEContext, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.vert_c_dim = vert_c_dim

        self.context_proj = nn.Linear(128, vert_c_dim)  
        # context are not added to value
        self.q = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        C_qk = C+self.vert_c_dim

        context = context.repeat(B//context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H*W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)

        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)

        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        x_qk = x_qk.reshape(B, _h, self.ws, _w, self.ws, C_qk).transpose(2, 3)

        v = self.v(x).reshape(
            B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]

        coords = coords_grid(B, self.ws, self.ws).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk).view(B, self.ws, self.ws, C_qk)   
        # coords_enc:   B, ws, ws, C
        # x:            B, _h, _w, self.ws, self.ws, C
        x_qk = x_qk + coords_enc[:, None, None, :, :, :]

        q = self.q(x_qk).reshape(
            B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        k = self.k(x_qk).reshape(
            B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalSubSampleAttnRPEContext(nn.Module): 
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1, vert_c_dim=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.vert_c_dim = vert_c_dim
        self.context_proj = nn.Linear(128, vert_c_dim) 
        self.q = nn.Linear(dim+vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_key = nn.Conv2d(dim+vert_c_dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_value = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        C_qk = C + self.vert_c_dim
        H, W = size
        context = context.repeat(B//context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H*W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)
        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)
        pad_l = pad_t = 0
        pad_r = (self.sr_ratio - W % self.sr_ratio) % self.sr_ratio
        pad_b = (self.sr_ratio - H % self.sr_ratio) % self.sr_ratio
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        _, Hp, Wp, _ = x.shape
        padded_size = (Hp, Wp)
        padded_N = Hp*Wp
        x = x.view(B, -1, C)
        x_qk = x_qk.view(B, -1, C_qk)

        coords = coords_grid(B, *padded_size).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk)   
        # coords_enc:   B, Hp*Wp, C
        # x:            B, Hp*Wp, C
        q = self.q(x_qk + coords_enc).reshape(B, padded_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_key is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *padded_size)
            x_qk = x_qk.permute(0, 2, 1).reshape(B, C_qk, *padded_size)
            x = self.sr_value(x).reshape(B, C, -1).permute(0, 2, 1)
            x_qk = self.sr_key(x_qk).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x_qk = self.norm(x_qk)

        coords = coords_grid(B, padded_size[0] // self.sr_ratio, padded_size[1] // self.sr_ratio).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        # align the coordinate of local and global
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = self.k(x_qk + coords_enc).reshape(B, (padded_size[0] // self.sr_ratio)*(padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, (padded_size[0] // self.sr_ratio)*(padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class LocallyGroupedAttnRPE(nn.Module): 
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(LocallyGroupedAttnRPE, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        v = self.v(x).reshape(
            B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]

        coords = coords_grid(B, self.ws, self.ws).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C).view(B, self.ws, self.ws, C)   
        # coords_enc:   B, ws, ws, C
        # x:            B, _h, _w, self.ws, self.ws, C
        x = x + coords_enc[:, None, None, :, :, :]

        q = self.q(x).reshape(
            B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        k = self.k(x).reshape(
            B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)[0]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalSubSampleAttnRPE(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.sr_ratio - W % self.sr_ratio) % self.sr_ratio
        pad_b = (self.sr_ratio - H % self.sr_ratio) % self.sr_ratio
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        padded_size = (Hp, Wp)
        padded_N = Hp*Wp
        x = x.view(B, -1, C)

        coords = coords_grid(B, *padded_size).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)   
        # coords_enc:   B, Hp*Wp, C
        # x:            B, Hp*Wp, C
        q = self.q(x + coords_enc).reshape(B, padded_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *padded_size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        coords = coords_grid(B, padded_size[0] // self.sr_ratio, padded_size[1] // self.sr_ratio).to(x.device) 
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        # align the coordinate of local and global
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = self.k(x + coords_enc).reshape(B, (padded_size[0] // self.sr_ratio)*(padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, (padded_size[0] // self.sr_ratio)*(padded_size[1] // self.sr_ratio), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(
            B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalSubSampleAttn(nn.Module):    ########
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=None, with_rpe=False, vert_c_dim=0, groupattention=False, cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if groupattention:
            assert with_rpe, "Not implementing groupattention without rpe"
            if vert_c_dim > 0:
                self.attn = GroupAttnRPEContext(dim, num_heads, attn_drop, drop, ws, cfg, vert_c_dim)
            else:
                self.attn = GroupAttnRPE(dim, num_heads, attn_drop, drop, ws, cfg)
        elif ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            if with_rpe:
                if vert_c_dim > 0:
                    self.attn = GlobalSubSampleAttnRPEContext(dim, num_heads, attn_drop, drop, sr_ratio, vert_c_dim)
                else:
                    self.attn = GlobalSubSampleAttnRPE(dim, num_heads, attn_drop, drop, sr_ratio)
            else:
                self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, drop, sr_ratio)
        else:
            if with_rpe:
                if vert_c_dim > 0:
                    self.attn = LocallyGroupedAttnRPEContext(dim, num_heads, attn_drop, drop, ws, vert_c_dim)
                else:
                    self.attn = LocallyGroupedAttnRPE(dim, num_heads, attn_drop, drop, ws)
            else:
                self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, size: Size_, context=None):
        x = x + self.drop_path(self.attn(self.norm1(x), size, context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x