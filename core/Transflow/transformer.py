#import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .mae_encoders import PyramidVisionTransformerV2#, vit_large, swin_large
from .checkpoint import load_checkpoint
from .encoder import Encoder
from .decoder import Decoder
from .extractor import BasicEncoder

class Transflow(nn.Module):
    def __init__(self, cfg):
        super(Transflow, self).__init__()
        self.cfg = cfg

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        if cfg.cnet == 'mae_vit':
            self.context_encoder = vit_large(pretrained=self.cfg.pretrain)
            self.context_encoder.load_state_dict(torch.load("mae_vit.pth", map_location='cpu'), strict=False)
            
        elif cfg.cnet == 'mae_swin':
            self.context_encoder = swin_large(pretrained=self.cfg.pretrain)
            self.context_encoder.load_state_dict(torch.load("mae_swin.pth", map_location='cpu'), strict=False)
    
        elif cfg.cnet == 'mae_pvt':
        
           self.context_encoder = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1])

           self.context_encoder.load_state_dict(torch.load("mae_pvt.pth", map_location='cpu'), strict=False)
        
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, image1, image2, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        cost_memory, cost_volume_for_softcorrmap = self.encoder(image1, image2, data, context)

        flow_predictions, softCorrMap = self.decoder(cost_memory, context, cost_volume_for_softcorrmap, data, flow_init=flow_init)

        return flow_predictions, softCorrMap

