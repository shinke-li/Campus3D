from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
import argparse
ROOT_DIR = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, ROOT_DIR)
from datetime import datetime
from utils import logger
from engine import config
from importlib import import_module
from dataset.reader import read_h_matrix_file_list
from dataset.loader import TorchDataset, TorchDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, default='configs/sem_seg_default_block',
                    help='config directory')
args = parser.parse_args()

# Load config
abs_cfg_dir = os.path.abspath(os.path.join(__file__, "../../{}".format(args.config)))
config.merge_cfg_from_dir(abs_cfg_dir)
cfg = config.CONFIG

#Load logger
logger.file_output('{}.log'.format(datetime.now().strftime('%Y_%m_%d_%I%M')),
                   level='info')
log = logger.LOG
log.info('Config DIR: {}\n PID: {}'.format(args.config, os.getpid()))
log.info('Config Details {}:'.format(cfg))

#load hierirachical relationship
HM = read_h_matrix_file_list(cfg.DATASET.DATA.H_MATRIX_LIST_FILE)


#Load train module
train_process = import_module("models." + cfg.MODEL + '.campusnet_train')
metric_cal = import_module('utils.metric')

#Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in cfg.DEVICES.GPU_ID])
#train_process.train(cfg, None, None, metric_cal, h_matrices=HM, logger=None)
#Load dataset
TRAIN_DATASET = TorchDataset("TRAIN_SET",
                            params=cfg.DATASET,
                            is_training=True,
                             )
TRAIN_LOADER = TorchDataLoader(dataset=TRAIN_DATASET,
                               batch_size=cfg.TRAIN.BATCH_SIZE,
                               num_workers=int(cfg.TRAIN.BATCH_SIZE / 2)
                               )
VALIDATION_DATASET = TorchDataset("VALIDATION_SET",
                                  params=cfg.DATASET,
                                  is_training=False,
                                  )
VALIDATION_LOADER = TorchDataLoader(dataset=VALIDATION_DATASET,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    num_workers=int(cfg.TRAIN.BATCH_SIZE / 2)
                                    )
train_logger = logger.gen_logger(os.path.join(cfg.TRAIN.OUTPUT_DIR, 'train.log'))
log.info('Output Dir: {}'.format(cfg.TRAIN.OUTPUT_DIR))
train_process.train(cfg, TRAIN_LOADER, VALIDATION_LOADER, metric_cal, h_matrices=HM, logger=train_logger)
