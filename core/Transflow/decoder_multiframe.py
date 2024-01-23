import loguru
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange

from utils.utils import coords_grid, bilinear_sampler, upflow8
from .attention import MultiHeadAttention, LinearPositionEmbeddingSine, ExpPositionEmbeddingSine
from timm.models.layers import DropPath

from .update import BasicUpdateBlock, GMAUpdateBlock
from .gma import Attention


def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init



class CrossAttention(nn.Module):
    def __init__(
        self,
        qk_dim,
        v_dim,
        query_token_dim,
        tgt_token_dim,
        add_flow_token=True,
        num_heads=8,
        attn_drop=0.,
        proj_drop=0.,
        drop_path=0.,
        dropout=0.,
        pe='linear',
        ):
        super(CrossAttention, self).__init__()
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5
        self.query_token_dim = query_token_dim
        self.pe = pe

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        (self.q, self.k, self.v) = (nn.Linear(query_token_dim, qk_dim,
                                    bias=True),
                                    nn.Linear(tgt_token_dim, qk_dim,
                                    bias=True),
                                    nn.Linear(tgt_token_dim, v_dim,
                                    bias=True))

        self.proj = nn.Linear(v_dim * 2, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = (DropPath(drop_path) if drop_path
                          > 0. else nn.Identity())

        self.ffn = nn.Sequential(nn.Linear(query_token_dim,
                                 query_token_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(query_token_dim,
                                 query_token_dim), nn.Dropout(dropout))
        self.add_flow_token = add_flow_token
        self.dim = qk_dim
        
    def forward(self, query, key, value, memory, query_coord, patch_size, size_h3w3):
        """
        Returns:
        - x (torch.Tensor): Output.
        - k (torch.Tensor): Key.
        - v (torch.Tensor): Value.
        """
        B, _, H1, W1 = query_coord.shape

        if key is None and value is None:
            key = self.k(memory)
            value = self.v(memory)

        # Reshape query coordinate
        query_coord = query_coord.contiguous()
        query_coord = query_coord.view(B, 2, -1).permute(0, 2, 1)[:,:,None,:].contiguous().view(B*H1*W1, 1, 2)

        # Encode query with position encoding
        if self.pe == 'linear':
            query_coord_enc = LinearPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == 'exp':
            query_coord_enc = ExpPositionEmbeddingSine(query_coord, dim=self.dim)

        short_cut = query

        query = self.norm1(query)
        
        if self.add_flow_token:
            q = self.q(query+query_coord_enc)
        else:
            q = self.q(query_coord_enc)

        # Compute key and value
        k, v = key, value

        x = self.multi_head_attn(q, k, v)
        x = self.proj(torch.cat([x, short_cut],dim=2))
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x, k, v
    
    
class DecoderLayer(nn.Module):
    def __init__(self, dim, cfg):
        super(DecoderLayer, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size 
        query_token_dim, tgt_token_dim = cfg.query_latent_dim, cfg.cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.cross_layer = CrossAttention(qk_dim, v_dim, query_token_dim, tgt_token_dim, add_flow_token=cfg.add_flow_token, dropout=cfg.dropout)

    def forward(self, query, key, value, memory, coords1, size, size_h3w3):
        x_global, k, v = self.cross_layer(query, key, value, memory, coords1, self.patch_size, size_h3w3)
        B, C, H1, W1 = size
        C = self.cfg.query_latent_dim
        x_global = x_global.view(B, H1, W1, C).permute(0, 3, 1, 2)
        return x_global, k, v
    
    
    
class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        dim = self.dim = cfg.query_latent_dim
        self.cfg = cfg

        self.flow_token_encoder = nn.Sequential(
            nn.Conv2d(81*cfg.cost_heads_num, dim, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1)
        )
        self.proj = nn.Conv2d(128, 256, 1)
        self.depth = cfg.decoder_depth
        self.decoder_layer = DecoderLayer(dim, cfg)
        
        if self.cfg.gma:
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.att = Attention(args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128)
        else:
            self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=128)
    
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def encode_flow(self, cost_maps, coords):
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        r = 4
        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid = coords.reshape(batch*h1*w1, 1, 1, 2)
        delta = delta.view(1, 2*r+1, 2*r+1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(cost_maps, coords)
        corr = corr.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
        return corr


    def forward(self, cost_memory12, context12, cost_volume_for_softcorrmap12, cost_memory23, context23, cost_volume_for_softcorrmap23, data={}, flow_init=None):

        cost_maps_12 = data['cost_maps_1to2']
        cost_maps_23 = data['cost_maps_2to3']
        coords0_12, coords1_12 = initialize_flow(context12)
        coords0_23, coords1_23 = initialize_flow(context23)


        N, fC, fH, fW = context12.shape
        corrMap12 = cost_volume_for_softcorrmap12.squeeze().view(N, fH*fW, -1)
        corrMap23 = cost_volume_for_softcorrmap23.squeeze().view(N, fH*fW, -1)
        
        softCorrMap12 = F.softmax(corrMap12, dim=2) * F.softmax(corrMap12, dim=1) # (N, fH*fW, fH*fW)
        softCorrMap23 = F.softmax(corrMap23, dim=2) * F.softmax(corrMap23, dim=1) # (N, fH*fW, fH*fW)


        #print("[Using warm start]")
        if flow_init is not None:
            coords1_12 = coords1_12 + flow_init
            coords1_23 = coords1_23 + flow_init


        flow_predictions = []
        softCorrMap = []

        context12, context23 = self.proj(context12), self.proj(context23)
        net_12, inp_12 = torch.split(context12, [128, 128], dim=1)
        net_23, inp_23 = torch.split(context23, [128, 128], dim=1)
        
        net_12 = torch.tanh(net_12)
        net_23 = torch.tanh(net_23)
        
        inp_12 = torch.relu(inp_12)
        inp_23 = torch.relu(inp_23)
        
        if self.cfg.gma:
            attention_12 = self.att(inp_12)
            attention_23 = self.att(inp_23)

        size = net_12.shape
        key_12, key_23, value_12, value_23 = None, None, None, None
        
        for idx in range(self.depth):
            coords1_12 = coords1_12.detach()
            coords1_23 = coords1_23.detach()
            
            cost_forward_12 = self.encode_flow_token(cost_maps_12, coords1_12)
            cost_forward_23 = self.encode_flow_token(cost_maps_23, coords1_23)

            query_12 = self.flow_token_encoder(cost_forward_12)
            query_23 = self.flow_token_encoder(cost_forward_23)
            
            query_12 = query_12.permute(0, 2, 3, 1).contiguous().view(size[0]*size[2]*size[3], 1, self.dim)
            query_23 = query_23.permute(0, 2, 3, 1).contiguous().view(size[0]*size[2]*size[3], 1, self.dim)
            
            cost_global_12, key_12, value_12 = self.decoder_layer(query_12, key_12, value_12, cost_memory12, coords1_12, size, data['H3W3'])
            cost_global_23, key_23, value_23 = self.decoder_layer(query_23, key_23, value_23, cost_memory23, coords1_23, size, data['H3W3'])
            

            if self.cfg.only_global:  
                corr_12 = cost_global_12
                corr_23 = cost_global_23
            else:
                corr_12 = torch.cat([cost_global_12, cost_forward_12], dim=1)
                corr_23 = torch.cat([cost_global_23, cost_forward_23], dim=1)


            flow_12 = coords1_12 - coords0_12
            flow_23 = coords1_23 - coords0_23
             
            if self.cfg.gma:
                net_12, up_mask_12, delta_flow_12 = self.update_block(net_12, inp_12, corr_12, flow_12, attention_12)
                net_23, up_mask_23, delta_flow_23 = self.update_block(net_23, inp_23, corr_23, flow_23, attention_23)
            else:
                net_12, up_mask_12, delta_flow_12 = self.update_block(net_12, inp_12, corr_12, flow_12)
                net_23, up_mask_23, delta_flow_23 = self.update_block(net_23, inp_23, corr_23, flow_23)

            # flow = delta_flow
            coords1_12 = coords1_12 + delta_flow_12
            coords1_23 = coords1_23 + delta_flow_23
            
            flow_up_12 = self.upsample_flow(coords1_12 - coords0_12, up_mask_12)
            flow_up_23 = self.upsample_flow(coords1_23 - coords0_23, up_mask_23)
            
            
            flow_predictions.append(torch.stack([flow_up_12, flow_up_23], dim=1))
            
        softCorrMap.append(torch.stack([softCorrMap12, softCorrMap23], dim=1))
       

        if self.training:
            return flow_predictions, softCorrMap
        else:
            return flow_predictions[-1], torch.stack([coords1_12-coords0_12, coords1_23-coords0_23], dim=1)
