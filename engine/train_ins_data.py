from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

from utils import logger
from engine import config
from importlib import import_module
from dataset.hierachical_matrix import load, all_label_project
from dataset.loader import  TorchDataset, TorchDataLoader
from utils.h5 import open_index_h5
import numpy as np

cfg_dir = 'new_ins_seg_SGPN_block'

# Load config
abs_cfg_dir = os.path.abspath(os.path.join(__file__, "../../configs/{}".format(cfg_dir)))
config.merge_cfg_from_dir(abs_cfg_dir)
cfg = config.CONFIG

#Load logger
if not os.path.isdir(cfg.TRAIN.OUTPUT_DIR): os.mkdir(cfg.TRAIN.OUTPUT_DIR)
logger_file = os.path.join(cfg.TRAIN.OUTPUT_DIR, '{}_{}.log'.format(cfg_dir, os.getpid()))
logger.file_output(logger_file, level='info')
log = logger.LOG
log.info('Config:' + str(cfg))

#Load train module
train_process = import_module("models." + cfg.MODEL + '.campusnet_train')
#metric_cal = import_module('utils.metric')

#Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in cfg.DEVICES.GPU_ID])

#Training
if True:
    TRAIN_DATASET = TorchDataset(set_name='TRAIN_SET',
                                 params=cfg.DATASET,
                                 is_training=True,)
    TRAIN_LOADER = TorchDataLoader(dataset=TRAIN_DATASET,
                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                   num_workers=10,
                                   )
    #VALIDATION_DATASET = TorchDataset(cfg.DATASET.VALIDATION_SET,
                                     # params=cfg.DATASET,
                                     # is_training=False,)
    #VALIDATION_LOADER = TorchDataLoader(dataset=VALIDATION_DATASET,
                                       # batch_size=cfg.TRAIN.BATCH_SIZE,
                                       # num_workers=int(cfg.TRAIN.BATCH_SIZE / 2))

    #log.info('len_per_batch{}'.format(len(TRAIN_DATASET)/150))
    train_logger = logger.gen_logger(os.path.join(cfg.TRAIN.OUTPUT_DIR, 'train.log'))
    train_process.train(cfg, TRAIN_LOADER, None, logger=train_logger)#metric_cal, project_matrices=h_project, matrices=matrices)

