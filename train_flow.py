from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
import evaluate_Flow as evaluate
import core.datasets as datasets
from core.loss import sequence_loss
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger
from core.Transflow import build_transformer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
from tqdm import tqdm
import time



try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


def fetch_optimizer(model, cfg):
    """ Create the optimizer and learning rate scheduler """
    name = cfg.optimizer
    lr = cfg.canonical_lr

    if name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.adam_decay, eps=cfg.epsilon)
    elif name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.adamw_decay, eps=cfg.epsilon)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")


    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, cfg.num_steps+100,
                                            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler
    
    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    model = nn.DataParallel(build_transformer(cfg), device_ids=args.gpus)

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)


    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    add_noise = False

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            images, flows, valids = [x.cuda() for x in data_blob]
            
            # modify to load arbitary number of images
            image1 = images[:, 0]
            image2 = images[:, 1]
            image3 = images[:, 2]
            
            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                image3 = (image2 + stdv * torch.randn(*image3.shape).cuda()).clamp(0.0, 255.0)

            output = {}
            flow_predictions, softCorrMap = model(images, output)
            loss, metrics = sequence_loss(flow_predictions, flows, valids, softCorrMap, image1, image2, cfg)    ###
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)
            


            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                print('=======Total_steps %d, Eval now!!!' % total_steps)

                if cfg.validation == 'chairs':
                    results_out = evaluate.validate_chairs(model.module)
                    print("Eval on Chairs!!! EPE is %f"  % (results_out['chairs']))
                    results.update(results_out)
                
                elif cfg.validation == 'sintel':
                    results_out = evaluate.validate_sintel(model.module)
                    print('Eval on Clean!!!  EPE: %f' % results_out['clean'])
                    print('Eval on Final!!!  EPE: %f' % results_out['final'])                        
                    results.update(results_out)
                elif cfg.validation == 'kitti':
                    results_out = evaluate.validate_kitti(model.module)
                    print("Eval on KITTI!!! epe is %f, f1 is %f" % (results_out['kitti-epe'], results_out['kitti-f1']))
                    results.update(results_out)


                logger.write_dict(results)
                
                model.train()
            
            total_steps += 1
            

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/{cfg.stage}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='chairs', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training", default='chairs') 
    parser.add_argument('--validation', type=str, default='chairs')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'autoflow':
        from configs.autoflow import get_cfg
    elif args.stage == 'sintel_ft':
        from configs.sintel_ft import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)


    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')


    train(cfg)
