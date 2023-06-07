import sys
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submission import cfg
from utils import flow_viz
from utils import frame_utils
import cv2
import math
import os.path as osp
from core.Transflow import build_transformer
from utils.utils import InputPadder
import itertools


def compute_adaptive_image_size(image_size):
    target_size = [360, 800]
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size


def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size):
    print("Reading image...")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


    dirname = osp.dirname(fn1)
    filename = osp.splitext(osp.basename(fn1))[0]

    viz_dir = osp.join(viz_root_dir, dirname)
    if not osp.exists(viz_dir):
        os.makedirs(viz_dir)

    viz_fn = osp.join(viz_dir, filename + '.png')
    return image1, image2, viz_fn


def build_model():
    model = torch.nn.DataParallel(build_transformer(cfg))
    model.load_state_dict(torch.load(cfg.model), strict=True)

    model.cuda()
    model.eval()

    return model


def infer(root_dir, viz_root_dir, model, img_pairs, keep_size):
    for img_pair in img_pairs:
        fn1, fn2 = img_pair
        image1, image2, viz_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size)

        image1, image2 = image1[None].cuda(), image2[None].cuda()
        
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)

        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()        
        
        flow_img = flow_viz.flow_to_image(flow)
        
        cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])
        
        
def process_inference(demo_dir):
    img_pairs = []
    image_list = sorted(glob(osp.join(demo_dir, '*.png')))
    for i in range(len(image_list)-1):
        img_pairs.append((image_list[i], image_list[i+1]))

    return img_pairs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--demo_dir', default='demo_frames/')
    parser.add_argument('--viz_root', default='./demo_viz_output/')
    parser.add_argument('--keep_size', action='store_true')     # keep the ori image size or resized

    args = parser.parse_args()

    root_dir = args.root_dir
    viz_root_dir = args.viz_root

    model = build_model()

    img_pairs = process_inference(args.demo_dir)

    with torch.no_grad():
        infer(root_dir, viz_root_dir, model, img_pairs, args.keep_size)