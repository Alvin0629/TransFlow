import time
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_cfg(cfg):
    
    # Define log directory path
    log_dir = f'logs/{cfg.name}/{cfg.transformer}/'

    # Get current time and format it
    now = time.localtime()
    now_time = '{:02d}_{:02d}_{:02d}_{:02d}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    log_dir += '(' + now_time + ')'
    
    cfg.log_dir = log_dir
    os.makedirs(log_dir)

    # Copy necessary files to log directory
    try:
        shutil.copytree('configs', f'{log_dir}/configs')
        shutil.copytree('core/Transflow', f'{log_dir}/Transflow')
    except Exception as e:
        logger.error("Error copying files to log directory: %s", e)
        raise
    else:
        logger.info("Files copied to log directory successfully.")