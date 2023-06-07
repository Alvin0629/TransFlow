import torch
def build_transformer(cfg):
    name = cfg.transformer 
    if name == 'Transflow':
        from .transformer import Transflow
    else:
        raise ValueError(f"{name} is not a valid architecture!")

    return Transflow(cfg[name])