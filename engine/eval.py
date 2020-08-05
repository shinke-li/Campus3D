from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
from utils import logger
from engine import config
from importlib import import_module
from dataset.loader import TorchDataset, TorchDataLoader
from dataset.reader import read_h_matrix_file_list
from utils.interpolation import interpolate
from utils.label_fusion import heirarchical_ensemble
from utils.metric import IouMetric, AccuracyMetric, HierarchicalConsistency
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config', type=str, default='sem_seg_default_block',
                    help='config directory')
parser.add_argument('-s', '--set', type=str, default='TEST_SET',
                    help='evaluation set: TEST_SET/VALIDATION_SET')
parser.add_argument('-ckpt', '--check_point', type=str, default='epoch108_model.ckpt',
                    help='check point file name')
args = parser.parse_args()

eval_set = args.set

# Load config
abs_cfg_dir = os.path.abspath(os.path.join(__file__, "../../configs/{}".format(args.config)))
config.merge_cfg_from_dir(abs_cfg_dir)
cfg = config.CONFIG

#Load logger
logger.file_output('eval_{}_{}.log'.format(eval_set,datetime.now().strftime('%Y_%m_%d_%I%M')),
                   level='info')
log = logger.LOG
log.info('Config DIR: {}\n PID: {}'.format(args.config, os.getpid()))
log.info('Config Details {}:'.format(cfg))

#load hierirachical relationship
HM = read_h_matrix_file_list(cfg.DATASET.DATA.H_MATRIX_LIST_FILE)


#Load train module
test_process = import_module("models." + cfg.MODEL + '.campusnet_test')
metric_cal = import_module('utils.metric')

#Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in cfg.DEVICES.GPU_ID])


log.info('Eval area:{}'.format(eval_set))
if True:
    TEST_DATASET = TorchDataset(eval_set,
                                params=cfg.DATASET,
                                is_training=False,)
    TEST_LOADER = TorchDataLoader(dataset=TEST_DATASET,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.BATCH_SIZE)
    log.info('Eval size:{}'.format(TEST_DATASET.labels[0].shape))

    points, logits = test_process.test(cfg, TEST_LOADER, log, args.check_point)

    log.info('Heirarchical ensemble')
    path_label = heirarchical_ensemble(logits, HM, weight=np.full((5,), 1.0))

    log.info('Interpolation')
    D, I = interpolate(sparse_points=points, dense_points=TEST_DATASET.points[0], GPU_id=0)

    

    log.info('Cal IoU/OA MT')
    pred_labels = []
    for i in range(len(logits)):
        log.info('IoU {}'.format(i))
        tmp_label = np.argmax(logits[i], axis=1)
        new_label = tmp_label[I]
        iou = IouMetric.cal_iou(np.squeeze(new_label), TEST_DATASET.labels[0][...,i], label_range=list(range(logits[i].shape[-1])))
        
        iou_string = [str(layer_iou) for layer_iou in iou]
        iou_string = '\n'.join(iou_string)
        log.info(iou_string)
        oa = AccuracyMetric.cal_oa(pred=np.squeeze(new_label), target=TEST_DATASET.labels[0][...,i])
        log.info('OA {}:{}'.format(i, oa))
        pred_labels.append(new_label)
    
    labels = np.asarray(pred_labels).transpose()
    cr = HierarchicalConsistency.cal_consistency_rate(HM, np.squeeze(labels))
    log.info('Cal consistent rate MT: {}'.format(cr))

    
    log.info('Cal IoU/OA HE')
    pred_labels = []
    for i in range(len(logits)):
        log.info('IoU {}'.format(i))
        new_label = path_label[..., i][I]
        iou = IouMetric.cal_iou(np.squeeze(new_label), TEST_DATASET.labels[0][...,i], label_range=list(range(logits[i].shape[-1])))
        iou_string = [str(layer_iou) for layer_iou in iou]
        iou_string = '\n'.join(iou_string)
        log.info(iou_string)
        oa = AccuracyMetric.cal_oa(pred=np.squeeze(new_label), target=TEST_DATASET.labels[0][...,i])
        log.info('OA {}:{}'.format(i, oa))
        pred_labels.append(new_label)

    labels = np.asarray(pred_labels).transpose()
    cr = HierarchicalConsistency.cal_consistency_rate(HM, np.squeeze(labels))
    log.info('Cal consistent rate HE: {}'.format(cr))
