from yacs.config import CfgNode as CN
cfg = CN()

cfg.name = ''
cfg.suffix ='submission'
cfg.gamma = 0.8
cfg.max_flow = 400
cfg.batch_size = 3
cfg.sum_freq = 100
cfg.val_freq = 10000
cfg.image_size = [360, 800]
cfg.add_noise = False

cfg.transformer = 'Transflow'
cfg.model = 'checkpoints/demo.pth'

cfg.use_matching_loss = True
cfg.pos_weight = 1.0
cfg.net_weight = 1.0

# Transflow
cfg.Transflow = CN()
cfg.Transflow.pe = 'linear'
cfg.Transflow.dropout = 0.0 
cfg.Transflow.encoder_latent_dim = 128
cfg.Transflow.query_latent_dim = 64
cfg.Transflow.cost_latent_input_dim = 64
cfg.Transflow.cost_latent_token_num = 8
cfg.Transflow.cost_latent_dim = 128
cfg.Transflow.motion_feature_dim = 209
cfg.Transflow.arc_type = 'Transflow'
cfg.Transflow.cost_heads_num = 1
# encoder
cfg.Transflow.pretrain = True
cfg.Transflow.concat = False
cfg.Transflow.encoder_depth = 3
cfg.Transflow.patch_size = 4
cfg.Transflow.patch_embed = 'single'
cfg.Transflow.gma = "GMA"
cfg.Transflow.rm_res = True
cfg.Transflow.vert_c_dim = 64
cfg.Transflow.cost_encoder_res = True
cfg.Transflow.pwc_aug = False
cfg.Transflow.cnet = 'mae_pvt'
cfg.Transflow.fnet = 'mae_pvt'
cfg.Transflow.no_sc = False
cfg.Transflow.add_flow_token = True

# decoder
cfg.Transflow.decoder_depth = 32

### TRAINER
cfg.trainer = CN()
cfg.trainer.scheduler = 'OneCycleLR'
cfg.trainer.optimizer = 'adamw'
cfg.trainer.canonical_lr = 8.0e-5 
cfg.trainer.adamw_decay = 1e-5
cfg.trainer.clip = 1.0
cfg.trainer.num_steps = 300000
cfg.trainer.epsilon = 1e-8
cfg.trainer.anneal_strategy = 'linear'
def get_cfg():
    return cfg.clone()

