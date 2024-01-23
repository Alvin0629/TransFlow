import torch
from utils.utils import create_meshgrid, flow_warp
import numpy as np

MAX_FLOW = 400


@torch.no_grad()
def compute_supervision_coarse(flow, occlusions, scale: int):
    N, _, H, W = flow.shape
    Hc, Wc = int(np.ceil(H / scale)), int(np.ceil(W / scale))

    occlusions_c = occlusions[:, :, ::scale, ::scale]
    flow_c = flow[:, :, ::scale, ::scale] / scale
    occlusions_c = occlusions_c.reshape(N, Hc * Wc)

    grid_c = create_meshgrid(Hc, Wc, False, device=flow.device).reshape(1, Hc * Wc, 2).repeat(N, 1, 1)
    warp_c = grid_c + flow_c.permute(0, 2, 3, 1).reshape(N, Hc * Wc, 2)
    warp_c = warp_c.round().long()

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)

    occlusions_c[out_bound_mask(warp_c, Wc, Hc)] = 1
    warp_c = warp_c[..., 0] + warp_c[..., 1] * Wc

    b_ids, i_ids = torch.split(torch.nonzero(occlusions_c == 0), 1, dim=1)
    conf_matrix_gt = torch.zeros(N, Hc * Wc, Hc * Wc, device=flow.device)
    j_ids = warp_c[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1

    return conf_matrix_gt
    
def compute_coarse_loss(conf, conf_gt, cfg):
    c_pos_w, c_neg_w = cfg.POS_WEIGHT, cfg.NEG_WEIGHT
    pos_mask, neg_mask = conf_gt == 1, conf_gt == 0

    if cfg.COARSE_TYPE == 'cross_entropy':
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        loss_pos = -torch.log(conf[pos_mask])
        loss_neg = -torch.log(1 - conf[neg_mask])

        return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
    elif cfg.COARSE_TYPE == 'focal':
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        alpha = cfg.FOCAL_ALPHA
        gamma = cfg.FOCAL_GAMMA
        loss_pos = -alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
        loss_neg = -alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
        return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
    else:
        raise ValueError('Unknown coarse loss: {type}'.format(type=cfg.COARSE_TYPE))

    

def sequence_loss(flow_preds, flow_gt, valid, softCorrMap, image1, image2, cfg):
    gamma = cfg.gamma
    max_flow = cfg.max_flow

    n_predictions = len(flow_preds)    

    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    flow_gt = flow_gt[:, 0]
    softCorrMap = softCorrMap[0][:, 0]

    valid = valid[:, 0] 

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i][:, 0]  - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1][:, 0]  - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    ####  Occlusion Loss ####
    img_2back1 = flow_warp(image2, flow_gt)
    occlusionMap = (image1 - img_2back1).mean(1, keepdims=True) #(N, H, W)
    occlusionMap = torch.abs(occlusionMap)
    occlusionMap = occlusionMap.float().squeeze()

    if cfg.use_matching_loss:
        img_2back1 = flow_warp(image2, flow_gt)
        occlusionMap = (image1 - img_2back1).mean(1, keepdims=True) #(N, H, W)
        occlusionMap = torch.abs(occlusionMap) > 20
        occlusionMap = occlusionMap.float()

        conf_matrix_gt = compute_supervision_coarse(flow_gt, occlusionMap, 8) # 8 from RAFT downsample   

        cfg.POS_WEIGHT = 1
        cfg.NEG_WEIGHT = 1
        cfg.FOCAL_ALPHA = 0.25
        cfg.FOCAL_GAMMA = 2.0
        cfg.COARSE_TYPE = 'cross_entropy'
        match_loss = compute_coarse_loss(softCorrMap, conf_matrix_gt, cfg)

        flow_loss = flow_loss + 0.01 * match_loss 
    
    return flow_loss, metrics

