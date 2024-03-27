from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix ='kitti'
_CN.gamma = 0.85
_CN.max_flow = 400
_CN.batch_size = 8
_CN.sum_freq = 200
_CN.val_freq = 5000
_CN.image_size = [432, 960]
_CN.add_noise = True
_CN.critical_params = []

_CN.transformer = 'Transflow'
_CN.model = None 
_CN.restore_ckpt = "checkpoints/sintel.pth"

_CN.use_matching_loss = True
_CN.POS_WEIGHT = 1
_CN.NEG_WEIGHT = 1
_CN.FOCAL_ALPHA = 0.25
_CN.FOCAL_GAMMA = 2.0
_CN.COARSE_TYPE = 'cross_entropy'

# Transflow
_CN.Transflow = CN()
_CN.Transflow.pe = 'linear'
_CN.Transflow.dropout = 0.0 
_CN.Transflow.encoder_latent_dim = 128 
_CN.Transflow.query_latent_dim = 64
_CN.Transflow.cost_latent_input_dim = 64
_CN.Transflow.cost_latent_token_num = 8
_CN.Transflow.cost_latent_dim = 128
_CN.Transflow.motion_feature_dim = 209 
_CN.Transflow.arc_type = 'Transflow'
_CN.Transflow.cost_heads_num = 1
# encoder
_CN.Transflow.pretrain = True
_CN.Transflow.context_concat = False
_CN.Transflow.encoder_depth = 3
_CN.Transflow.feat_cross_attn = False
_CN.Transflow.patch_size = 4
_CN.Transflow.patch_embed = 'single'
_CN.Transflow.gma = "GMA"
_CN.Transflow.rm_res = True
_CN.Transflow.vert_c_dim = 64
_CN.Transflow.cost_encoder_res = True
_CN.Transflow.pwc_aug = False
_CN.Transflow.cnet = 'mae_pvt'
_CN.Transflow.fnet = 'mae_pvt'
_CN.Transflow.no_sc = False
_CN.Transflow.only_global = False
_CN.Transflow.add_flow_token = True
_CN.Transflow.use_mlp = False
_CN.Transflow.vertical_conv = False

# decoder
_CN.Transflow.decoder_depth = 12
_CN.Transflow.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']


### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 15e-5
_CN.trainer.adamw_decay = 1e-5
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 80000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
