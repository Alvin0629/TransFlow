# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import copy
import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

# Modify to arbitrary number of frames
class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, vkitti2=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.vkitti2 = vkitti2

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img3 = frame_utils.read_gen(self.image_list[index][2])

            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img3 = np.array(img3).astype(np.uint8)[..., :3]

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            img3 = torch.from_numpy(img3).permute(2, 0, 1).float()

            return torch.stack([img1, img2, img3]), self.extra_info[index]


        index = index % len(self.image_list)
        
        valid1 = None
        valid2 = None
        
        if self.sparse:
            if self.vkitti2:
                flow1, valid1 = frame_utils.read_vkitti2_flow(self.flow_list[index][0])
                flow2, valid2 = frame_utils.read_vkitti2_flow(self.flow_list[index][1])

            else:
                flow1, valid1 = frame_utils.readFlowKITTI(self.flow_list[index][0])  # [H, W, 2], [H, W]
                flow2, valid2 = copy.deepcopy(flow1) * 0., copy.deepcopy(valid1) * 0.

        else:
            flow1 = frame_utils.read_gen(self.flow_list[index][0])
            flow2 = frame_utils.read_gen(self.flow_list[index][1])
            

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])

        flow1 = np.array(flow1).astype(np.float32)
        flow2 = np.array(flow2).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img3 = np.array(img3).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
            img3 = np.tile(img3[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
            img3 = img3[..., :3]


        if self.augmentor is not None:
            if self.sparse:
                img1, img2, img3, flow1, flow2, valid1, valid2 = self.augmentor(img1, img2, img3, flow1, flow2, valid1, valid2)
                
            else:
                img1, img2, img3, flow1, flow2 = self.augmentor(img1, img2, img3, flow1, flow2)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img3 = torch.from_numpy(img3).permute(2, 0, 1).float()
        
        flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
        flow2 = torch.from_numpy(flow2).permute(2, 0, 1).float()

        if valid1 is not None and valid2 is not None:
            valid1 = torch.from_numpy(valid1)
            valid2 = torch.from_numpy(valid2) * 0
        else:
            valid1 = (flow1[0].abs() < 1000) & (flow1[1].abs() < 1000)
            valid2 = (flow2[0].abs() < 1000) & (flow2[1].abs() < 1000)

        return torch.stack([img1, img2, img3]), torch.stack([flow1, flow2]),  torch.stack([valid1.float(), valid2.float()])


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='Transflow/datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-3):
                self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                flow_list = sorted(glob(osp.join(flow_root, scene, '*.flo')))
                for i in range(len(flow_list)-2):
                    self.flow_list += [ [flow_list[i], flow_list[i+1]] ]




class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='Transflow/datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        
        identity_flow = osp.join('Transflow/datasets/FlyingChairs_release', 'identity.flo')

        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)-1):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ [flows[i], identity_flow] ]
                self.image_list += [ [images[2*i], images[2*i+1], images[2*i+1]] ]



class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='Transflow/datasets/FlyingThings3D', dstype='frames_cleanpass', split='training'):
        super(FlyingThings3D, self).__init__(aug_params)

        split_dir = 'TRAIN' if split == 'training' else 'TEST'
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'{split_dir}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])


                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )

                    if direction == 'into_future':
                        for i in range(len(flows)-2):
                            if direction == 'into_future':
                                self.image_list += [ [images[i], images[i+1], images[i+2]] ]
                                self.flow_list += [ [flows[i], flows[i+1]] ]

                    elif direction == 'into_past':
                        for i in range(len(flows)-3):    
                            if direction == 'into_past':
                                self.image_list += [ [images[i+2], images[i+1], images[i]] ]
                                self.flow_list += [ [flows[i+2], flows[i+1]] ]
                
      


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='Transflow/datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True


        identity_flow = osp.join('Transflow/datasets/KITTI', 'identity.flo')
        
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2, img2] ]

        if split == 'training':
            flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            for i in range(len(flow_list)):
                self.flow_list += [[flow_list[i], identity_flow]]




class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='Transflow/datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-2):
                self.flow_list += [ [flows[i], flows[i+1]] ]
                self.image_list += [ [images[i], images[i+1], images[i+2]] ]

            seq_ix += 1




class VKITTI2(FlowDataset):
    def __init__(self, aug_params=None,
                 root='Transflow/datasets/VKITTI2',
                 ):
        super(VKITTI2, self).__init__(aug_params, sparse=True, vkitti2=True,
                                      )

        data_dir = root

        scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

        for scene in scenes:
            scene_dir = os.path.join(data_dir, scene)

            types = os.listdir(scene_dir)

            for scene_type in types:
                type_dir = os.path.join(scene_dir, scene_type)

                imgs = sorted(glob(os.path.join(type_dir, 'frames', 'rgb', 'Camera_0', '*.jpg')))

                flows_fwd = sorted(glob(os.path.join(type_dir, 'frames', 'forwardFlow', 'Camera_0', '*.png')))
                flows_bwd = sorted(glob(os.path.join(type_dir, 'frames', 'backwardFlow', 'Camera_0', '*.png')))

                assert len(imgs) == len(flows_fwd) + 1 and len(imgs) == len(flows_bwd) + 1

                for i in range(len(imgs) - 2):
                    # forward
                    self.image_list += [[imgs[i], imgs[i + 1], imgs[i + 2]]]
                    self.flow_list += [[flows_fwd[i], flows_fwd[i + 1]]]

                    # backward
                    self.image_list += [[imgs[i + 2], imgs[i + 1], imgs[i]]]
                    self.flow_list += [[flows_bwd[i+1], flows_bwd[i]]]




def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset


    elif args.stage == 'sintel_ft':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        train_dataset = sintel_clean + 2 * sintel_final



    elif args.stage == 'sintel':
    
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')   
             

        if TRAIN_DS == 'C+T+K+S+H':
            print('Correct! using C+T+K+S+H')
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            vkitti2 = VKITTI2({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})  # 42420
            
            train_dataset = 100*sintel_clean + 120*sintel_final + 200*kitti + 5*hd1k + things + vkitti2 

        elif TRAIN_DS == 'C+T+K/S':
            print('Wrong! using C+T+K/S')
            train_dataset = 100*sintel_clean + 100*sintel_final + things 
            

    elif args.stage == 'kitti':
        print('Correct! using KITTI')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
