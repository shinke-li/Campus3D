from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.newcollections import AttrDict
from ast import literal_eval

import logging
import os
import yaml
import copy
import six
import glob
import numpy as np

logger = logging.getLogger(__name__)
__cfg = AttrDict()

CONFIG = __cfg

__cfg.MODEL = ""
# Train setting

__cfg.AUTO_ALLOW_EXTRA_KEY=False

__cfg.TRAIN = AttrDict()
# TRAIN.SETTING => train parameters


__cfg.TRAIN.BATCH_SIZE = 1
__cfg.TRAIN.MAX_EPOCH = 150
__cfg.TRAIN.OUTPUT_DIR = ''
__cfg.TRAIN.PRETRAINED_MODEL_PATH = ''
__cfg.TRAIN.LOSS_WEIGHTS = [1.0]
__cfg.TRAIN.CONSISTENCY_LOSS = False
__cfg.TRAIN.CONSISTENCY_WEIGHTS = [1.0]

__cfg.TRAIN.SETTING = AttrDict()



__cfg.DEVICES = AttrDict()
__cfg.DEVICES.GPU_ID = [0]

# Dataset loader setting
__cfg.DATASET = AttrDict()

__cfg.DATASET.DATA = AttrDict()
__cfg.DATASET.DATA.DATA_LIST_FILE = ""
__cfg.DATASET.DATA.LABEL_NUMBER = []
__cfg.DATASET.DATA.H_MATRIX_LIST_FILE = ""
__cfg.DATASET.DATA.USE_COLOR = True
__cfg.DATASET.DATA.COLOR_CHANNEL = ()

__cfg.DATASET.DATA.LABEL_NUMBER = []
__cfg.DATASET.DATA.REMOVE_ZERO_LABEL = True

# dataset sampler choice
__cfg.DATASET.SAMPLE = AttrDict()
__cfg.DATASET.SAMPLE.RANDOM_SCENE_CHOOSE = True  # False means traversing
__cfg.DATASET.SAMPLE.SAMPLER_TYPE = "BlockSampler"
__cfg.DATASET.SAMPLE.RETURN_INDEX = False
__cfg.DATASET.SAMPLE.RANDOM_SEED_BASIS = 0
__cfg.DATASET.SAMPLE.LABEL_WEIGHT_POLICY = 'ones'

# dataset sampler setting
__cfg.DATASET.SAMPLE.SETTING = AttrDict()
__cfg.DATASET.SAMPLE.SETTING.MODIFY_TYPE = ['block_centeralization',]
                                     #'global_normalization',]
_EXTRA_KEYS = set(
[
    'DATASET.SAMPLE.SETTING.NUM_POINTS_PER_SAMPLE',
    'DATASET.SAMPLE.SETTING.BOX_SIZE_X',
    'DATASET.SAMPLE.SETTING.BOX_SIZE_Y',
    'DATASET.SAMPLE.SETTING.SLIDING_RATIO',
    'DATASET.SAMPLE.SETTING.SPARSE_THRESH',
    'DATASET.SAMPLE.SETTING.ORIGIN_SAMPLER_TYPE',
    'DATASET.SAMPLE.SETTING.IGNORE_FINE_BOUNDS',


    'DATASET.SAMPLE.SETTING.KNN_MODULE',
    'DATASET.SAMPLE.SETTING.MAX_WORKERS',
'DATASET.SAMPLE.SETTING.OVERLAP_RATIO',


    'TRAIN.SETTING.MARGIN_DIFF',
    'TRAIN.SETTING.MARGIN_SAME',

    'TRAIN.SETTING.RADIUS'
]

)

def load_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))
    return yaml_cfg

def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    yaml_cfg = load_yaml(cfg_filename)
    _merge_a_into_b(yaml_cfg, __cfg)

def merge_cfg_from_dir(cfg_dirname):
    """Load a yaml config file and merge it into the global config."""
    for cfg_filename in glob.glob(os.path.join(cfg_dirname, '*.yaml')):
        yaml_cfg = load_yaml(cfg_filename)
        _merge_a_into_b(yaml_cfg, __cfg)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __cfg)

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k


        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        # a must specify keys that are in b
        if k not in b:
            if not _key_is_extra_setting(full_key) and not __cfg.AUTO_ALLOW_EXTRA_KEY:
                raise KeyError('Non-existent config key: {}'.format(full_key))
        else:
            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v

def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v

def _key_is_extra_setting(full_key):
    if full_key in _EXTRA_KEYS:
        logger.info(
            'Extra config key (ignoring): {}'.format(full_key)
        )
        return True
    return False

def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a