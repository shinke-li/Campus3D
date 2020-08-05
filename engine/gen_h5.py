from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
from engine import config
from dataset.loader import TorchDataset, TorchDataLoader
from utils.h5 import open_index_h5


import numpy as np
'''
#######################################
from utils.newcollections import AttrDict
__cfg = AttrDict()

cfg = __cfg


# Train setting

__cfg.TRAIN = AttrDict()
# TRAIN.SETTING => train parameters

__cfg.TRAIN.MODEL = ""
__cfg.TRAIN.BATCH_SIZE = 1

__cfg.TRAIN.SETTING = AttrDict()

# Dataset loader setting
__cfg.DATASET = AttrDict()

# dataset sampler choice
__cfg.DATASET.SAMPLER_TYPE = "BlockSampler"
# dataset split (file name)
__cfg.DATASET.TRAIN_SET = []
__cfg.DATASET.VALIDATION_SET = []
__cfg.DATASET.TEST_SET = []
# dataset load path
__cfg.DATASET.DATA_PATH = ""
__cfg.DATASET.LABEL_PATH = ""
__cfg.DATASET.LABEL_FILE_POSTFIXES = []

__cfg.DATASET.RANDOM_SEED_BASIS = 0
__cfg.DATASET.RANDOM_SCENE_CHOOSE = True  # False means traversing
__cfg.DATASET.USE_COLOR = True
__cfg.DATASET.COLOR_CHANNEL = [3, 4, 5]
__cfg.DATASET.LABEL_NUMBER = []
__cfg.DATASET.LABEL_WEIGHT_POLICY = 'ones'
__cfg.DATASET.CAL_BOUNDS = True
__cfg.DATASET.REMOVE_ZERO_LABEL = True

# for block sampler accelerating
__cfg.DATASET.SORTED_FLAG = True

# return point index from sampler
__cfg.DATASET.RETURN_INDEX = False

__cfg.DATASET.H5_DIR_PATH = ""
__cfg.DATASET.H5_FILE_PREFIX = ""

# dataset sampler setting
__cfg.DATASET.SETTING = AttrDict()
from engine.config import _merge_a_into_b, load_yaml
import glob

def merge_cfg_from_dir(cfg_dirname):
    for cfg_filename in glob.glob(os.path.join(cfg_dirname, '*.yaml')):
        yaml_cfg = load_yaml(cfg_filename)
        _merge_a_into_b(yaml_cfg, __cfg)
merge_cfg_from_dir(cfg_f)
#######################################
'''

cfg_f = os.path.abspath(os.path.join(__file__, "../../configs/train_index"))
config.merge_cfg_from_dir(cfg_f)
__cfg = config.CONFIG
f_name = os.path.join(__cfg.DATASET.H5_DIR_PATH, __cfg.DATASET.H5_FILE_PREFIX + '.hdf5')

print(f_name)
with open_index_h5(f_name, 'write', __cfg.DATASET.SETTING.NUM_POINTS_PER_SAMPLE) as h5f:
    TRAIN_DATASET = TorchDataset(__cfg.DATASET.TRAIN_SET,
                                 params=__cfg.DATASET,
                                 is_training=True)
    TRAIN_LOADER = TorchDataLoader(dataset=TRAIN_DATASET,
                                   batch_size=__cfg.TRAIN.BATCH_SIZE,
                                   num_workers=__cfg.TRAIN.BATCH_SIZE
                                   )
    MAX_EPOCH = 150
    try:
        for epoch in range(MAX_EPOCH):
            print('INFO: EPOCH NUM-{}'.format(epoch))
            for batch_idx, data_ in enumerate(TRAIN_LOADER):
                if batch_idx % 100 == 0:
                    print("INFO: BATCH NUM-{}".format(batch_idx))
                scene_ind_batch, batch_data_ind, _, _ = data_
                if scene_ind_batch.shape[0] != __cfg.TRAIN.BATCH_SIZE:
                    print('WARNING: EPOCH-{}-BATCH-{}-BATCH DATA SIZE-{}'.format(
                        epoch, batch_idx, scene_ind_batch.shape[0]
                    ))
                h5f.append(scene_ind_batch.astype(np.uint8), batch_data_ind.astype(np.uint32))
    except Exception as e:
        print(e)

#with open_index_h5(f_name, 'read') as h5f:
   # print(len(h5f))
   # print(np.bincount(h5f.read()[0]))