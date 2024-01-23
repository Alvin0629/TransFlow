import loguru
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
from functools import partial
from timm.models.layers import Mlp, DropPath
from einops.layers.torch import Rearrange
from einops import rearrange
from .att import Block
from utils.utils import coords_grid
from .attention import BroadMultiHeadAttention, MultiHeadAttention, LinearPositionEmbeddingSine, ExpPositionEmbeddingSine
from .mae_encoders import PyramidVisionTransformerV2#, vit_large, swin_large
from .checkpoint import load_checkpoint

from typing import Optional, Tuple
from .extractor import BasicEncoder

Size_ = Tuple[int, int]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=1, embed_dim=64, pe='linear'):
        super().__init__()
        self.patch_size = patch_size
        self.dim = embed_dim
        self.pe = pe

        # assert patch_size == 8
        if patch_size == 8:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=6, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=6, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(embed_dim//2, embed_dim, kernel_size=6, stride=2, padding=2),
            )
        elif patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim//4, kernel_size=6, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(embed_dim//4, embed_dim, kernel_size=6, stride=2, padding=2),
            )
        else:
            print(f"patch size = {patch_size} is unacceptable.")

        self.ffn_with_coord = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1)
        )
        self.norm = nn.LayerNorm(embed_dim*2)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:
        B, C, H, W = x.shape    # C == 1

        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        x = self.proj(x)
        out_size = x.shape[2:] 

        patch_coord = coords_grid(B, out_size[0], out_size[1]).to(x.device) * self.patch_size + self.patch_size/2 # in feature coordinate space
        patch_coord = patch_coord.view(B, 2, -1).permute(0, 2, 1)
        if self.pe == 'linear':
            patch_coord_enc = LinearPositionEmbeddingSine(patch_coord, dim=self.dim)
        elif self.pe == 'exp':
            patch_coord_enc = ExpPositionEmbeddingSine(patch_coord, dim=self.dim)
        patch_coord_enc = patch_coord_enc.permute(0, 2, 1).view(B, -1, out_size[0], out_size[1])

        x_pe = torch.cat([x, patch_coord_enc], dim=1)
        x =  self.ffn_with_coord(x_pe)
        x = self.norm(x.flatten(2).transpose(1, 2))

        return x, out_size


class GroupVerticalSelfAttentionLayer(nn.Module):
    def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(GroupVerticalSelfAttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = 0.
        drop_rate = dropout
        attn_drop_rate=0.

        self.block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=ws, with_rpe=True, vert_c_dim=cfg.vert_c_dim, groupattention=True, cfg=self.cfg)

    def forward(self, x, size, context=None):
        x = self.block(x, size, context)

        return x

class VerticalSelfAttentionLayer(nn.Module):
    def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(VerticalSelfAttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        dpr = 0.
        drop_rate = dropout
        attn_drop_rate=0.

        self.local_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=ws, with_rpe=True, vert_c_dim=cfg.vert_c_dim)
        self.global_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr, sr_ratio=sr_ratio, ws=1, with_rpe=True, vert_c_dim=cfg.vert_c_dim)

    def forward(self, x, size, context=None):
        x = self.local_block(x, size, context)
        x = self.global_block(x, size, context)

        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num +=  np.prod(param.size())

        return num

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(SelfAttentionLayer, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(dim, num_heads)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = x
        x = self.norm1(x)

        q, k, v = self.q(x), self.k(x), self.v(x)

        x = self.multi_head_attn(q, k, v)

        x = self.proj(x)
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num +=  np.prod(param.size())

        return num



class CrossAttention(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super(CrossAttention, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, tgt_token):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = query
        query = self.norm1(query)

        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)

        x = self.multi_head_attn(q, k, v)

        x = short_cut + self.proj_drop(self.proj(x))

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class CostEmbed(nn.Module):
    def __init__(self, cfg):
        super(CostEmbed, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.patch_embed = PatchEmbed(in_chans=self.cfg.cost_heads_num, patch_size=self.patch_size, embed_dim=cfg.cost_latent_input_dim, pe=cfg.pe)
        
        self.depth = cfg.encoder_depth

        self.latent_tokens = nn.Parameter(torch.randn(1, cfg.cost_latent_token_num, cfg.cost_latent_dim))

        query_token_dim, tgt_token_dim = cfg.cost_latent_dim, cfg.cost_latent_input_dim*2
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.input_layer = CrossAttention(qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout)

        self.encoder_layers = nn.ModuleList([SelfAttentionLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout) for idx in range(self.depth)])

        self.vertical_encoder_layers = nn.ModuleList([VerticalSelfAttentionLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout) for idx in range(self.depth)])
        
        self.cost_scale_aug = None
        if ('cost_scale_aug' in cfg.keys()):
            self.cost_scale_aug = cfg.cost_scale_aug
            print("[Using cost_scale_aug: {}]".format(self.cost_scale_aug))

        self.temporal_att = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.att_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        

    def forward(self, cost_volume1to2, cost_volume2to3, data, context=None, direction=None):
        B, heads, H1, W1, H2, W2 = cost_volume1to2.shape
                       
        cost_maps_1to2 = cost_volume1to2.permute(0, 2, 3, 1, 4, 5).contiguous().view(B*H1*W1, self.cfg.cost_heads_num, H2, W2)
        cost_maps_2to3 = cost_volume2to3.permute(0, 2, 3, 1, 4, 5).contiguous().view(B*H1*W1, self.cfg.cost_heads_num, H2, W2)
       
        # Temp att
        Temporal_att = self.temporal_att(cost_maps_1to2)
        cost_maps = cost_maps_1to2 + Temporal_att * (self.att_conv(cost_maps_2to3))  
       
        if direction == '1to2':
            data['cost_maps_1to2'] = cost_maps
        elif direction == '2to3':
            data['cost_maps_2to3'] = cost_maps

        if self.cost_scale_aug is not None:
            scale_factor = torch.FloatTensor(B*H1*W1, self.cfg.cost_heads_num, H2, W2).uniform_(self.cost_scale_aug[0], self.cost_scale_aug[1]).cuda()
            cost_maps = cost_maps * scale_factor
        
        x, size = self.patch_embed(cost_maps)   # B*H1*W1, size[0]*size[1], C
        data['H3W3'] = size
        H3, W3 = size

        x = self.input_layer(self.latent_tokens, x)

        short_cut = x

        for idx, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = x.view(B, H1*W1, self.cfg.cost_latent_token_num, -1).permute(0, 2, 1, 3).reshape(B*self.cfg.cost_latent_token_num, H1*W1, -1)
            x = self.vertical_encoder_layers[idx](x, (H1, W1), context)
            x = x.view(B, self.cfg.cost_latent_token_num, H1*W1, -1).permute(0, 2, 1, 3).reshape(B*H1*W1, self.cfg.cost_latent_token_num, -1)

        if self.cfg.cost_encoder_res is True:
            x = x + short_cut
        return x
    

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        if cfg.fnet == 'mae_vit':
            self.feat_encoder = vit_large(pretrained=self.cfg.pretrain)
            self.feat_encoder.load_state_dict(torch.load("mae_vit.pth", map_location='cpu'), strict=False)
            
        elif cfg.fnet == 'mae_swin':
            self.feat_encoder = swin_large(pretrained=self.cfg.pretrain)
            self.feat_encoder.load_state_dict(torch.load("mae_swin.pth", map_location='cpu'), strict=False)
     
        elif cfg.fnet == 'mae_pvt':
            self.feat_encoder = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1])
    
            self.feat_encoder.load_state_dict(torch.load("mae_pvt.pth", map_location='cpu'), strict=False)
            
        elif cfg.fnet == 'basicencoder':
            self.feat_encoder = BasicEncoder(output_dim=256, norm_fn='instance')
        else:
            exit()

        self.channel_convertor = nn.Conv2d(cfg.encoder_latent_dim, cfg.encoder_latent_dim, 1, padding=0, bias=False)

        self.encoder = CostEmbed(cfg)

    def corr(self, fmap1, fmap2):

        batch, dim, ht, wd = fmap1.shape
        fmap1 = rearrange(fmap1, 'b (heads d) h w -> b heads (h w) d', heads=self.cfg.cost_heads_num)
        fmap2 = rearrange(fmap2, 'b (heads d) h w -> b heads (h w) d', heads=self.cfg.cost_heads_num)
        corr = einsum('bhid, bhjd -> bhij', fmap1, fmap2)
        corr = corr.permute(0, 2, 1, 3).view(batch*ht*wd, self.cfg.cost_heads_num, ht, wd)
        #corr = self.norm(self.relu(corr))
        corr = corr.view(batch, ht*wd, self.cfg.cost_heads_num, ht*wd).permute(0, 2, 1, 3)
        corr = corr.view(batch, self.cfg.cost_heads_num, ht, wd, ht, wd)

        return corr


    def corr_for_softcorrmap(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, -1).permute(0, 2, 1)        
        fmap2 = fmap2.view(batch, dim, -1) 
        corr = torch.matmul(fmap1, fmap2).view(batch, ht, wd, ht, wd) / (dim ** 0.5)  # [B, H, W, H, W]
        corr = corr.view(batch, self.cfg.cost_heads_num, ht, wd, ht, wd) 
        return corr


    def forward(self, images, data, context=None, direction=None):
        # modify to arbitary number of images
        imgs1 = images[:, 0, ...]
        imgs2 = images[:, 1, ...]
        imgs3 = images[:, 2, ...]

        feats1 = self.feat_encoder(imgs1)
        feats1 = self.channel_convertor(feats1)
        feats2 = self.feat_encoder(imgs2)
        feats2 = self.channel_convertor(feats2)
        feats3 = self.feat_encoder(imgs3)
        feats3 = self.channel_convertor(feats3)

        fmap1 = feats1.float()
        fmap2 = feats2.float()
        fmap3 = feats3.float()


        B, C, H, W = fmap1.shape
        size = (H, W)
        
        cost_volume_1to2 = self.corr(fmap1, fmap2)
        cost_volume_2to3 = self.corr(fmap2, fmap3)

        if direction == '1to2':
            x = self.encoder(cost_volume_1to2, cost_volume_2to3, data, context, direction)
        
        elif direction == '2to3':
            x = self.encoder(cost_volume_2to3, cost_volume_1to2, data, context, direction)
            

        return x, cost_volume_1to2, cost_volume_2to3