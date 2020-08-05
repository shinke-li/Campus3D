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
from dataset.loader import TorchDataset, TorchDataLoader
from utils.label_fusion import path_fusion
from utils.metric import IouMetric, AccuracyMetric
import utils.cal_knn as knn
import numpy as np
from dataset import hierachical_matrix as HM

STORE_LABEL = True

cfg_dir = 'sem_seg_default_block'

# Load config
abs_cfg_dir = os.path.abspath(os.path.join(__file__, "../../configs/{}".format(cfg_dir)))
config.merge_cfg_from_dir(abs_cfg_dir)
cfg = config.CONFIG

#Load logger
#logger.file_output('{}_{}.log'.format(cfg_dir, os.getpid()), level='info')
log = logger.LOG
log.info('Config:' + str(cfg))

#Load test module
test_process = import_module("models." + cfg.MODEL + '.campusnet_test')
metric_cal = import_module('utils.metric')

matrices = HM.load(cfg.DATASET.H_MATRIX_PATH, cfg.DATASET.H_MATRIX_POSTFIXES, add_zero=True)

#Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in cfg.DEVICES.GPU_ID])
test_area = cfg.DATASET.VALIDATION_SET
log.info('Test area:{}'.format(test_area))

if True:
    TEST_DATASET = TorchDataset(test_area,
                                params=cfg.DATASET,
                                is_training=False, )
    TEST_LOADER = TorchDataLoader(dataset=TEST_DATASET,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.BATCH_SIZE)
    dir_name = '/home/lixk/code/campusnet/data'
    if STORE_LABEL:
        STORE_PATH = dir_name
    prefixes = ['label2', 'label3', 'label5', 'label8', 'label14']

    logits = []
    Is = []
    for p in prefixes:
        lname = '_'.join([test_area[0], 'sem_seg', p, 'logits', 'single', 'head']) + '.npy'
        iname = '_'.join([test_area[0], 'sem_seg', p, 'indices']) + '.npy'
        l = os.path.join(dir_name, lname)
        i = os.path.join(dir_name,iname)
        logits.append(np.load(l))
        Is.append(np.load(i))
        print(Is[-1].shape)
    new_logits = [np.squeeze(lgs[i]) for i, lgs in zip(Is, logits)]
    for i in new_logits:
        print(i.shape)
    log.info('Calculate heirirachical score')

    direct_labels = [np.argmax(lgs, axis=1) for lgs in new_logits]
    print([l.shape for l in direct_labels])
    scores = HM.cal_HCS(np.asarray(direct_labels).transpose(), matrices)
    print(np.sum(scores>=1)/float(scores.shape[0]))
    #import sys
    #sys.exit(0)

    f_name = ['l2', 'l3', 'l5', 'l8', 'l14']

    matrices = [np.loadtxt(os.path.join(dir_name, f + '.csv'), delimiter=",")
                for f in f_name]
    log.info('Do label fusion')
    path_label = path_fusion(new_logits, matrices, zero_removed=False)
    print(path_label.shape)


    log_name = 'log_{}_{}.txt'.format('fusion', test_area[0])
    log.info('Result write in {}'.format(log_name))
    if STORE_LABEL:
        orgin_label = []
        ensemble_label = []
    with open(log_name, 'a') as record:

        log.info('Cal IoU ensemble')
        record.write('record ensemble\n')
        for i in range(len(logits)):
            new_label = path_label[..., i]
            print(TEST_DATASET.labels[0].shape)
            iou = IouMetric.cal_iou(np.squeeze(new_label + 1), TEST_DATASET.labels[0][...,i], label_range=list(range(logits[i].shape[-1])))
            if STORE_LABEL:
                ensemble_label.append(np.squeeze(new_label + 1))
            iou_string = [str(layer_iou) for layer_iou in iou]
            iou_string = '\n'.join(iou_string)
            record.write('IoU' + '\n')
            record.write(iou_string + '\n')
            record.write('OA' + '\n')
            oa = AccuracyMetric.cal_oa(np.squeeze(new_label + 1), TEST_DATASET.labels[0][..., i])
            record.write(str(oa) + '\n')
        record.write('record origin\n')
        log.info('Cal IoU origin')
        for i in range(len(logits)):
            tmp_label = np.argmax(new_logits[i], axis=1)
            iou = IouMetric.cal_iou(np.squeeze(tmp_label), TEST_DATASET.labels[0][..., i],
                                    label_range=list(range(logits[i].shape[-1])))
            if STORE_LABEL:
                orgin_label.append(np.squeeze(tmp_label))
            iou_string = [str(layer_iou) for layer_iou in iou]
            iou_string = '\n'.join(iou_string)
            record.write('IoU' + '\n')
            record.write(iou_string + '\n')
            record.write('OA' + '\n')
            oa = AccuracyMetric.cal_oa(np.squeeze(tmp_label), TEST_DATASET.labels[0][..., i])
            record.write(str(oa) + '\n')
if STORE_LABEL:
    orgin_label = np.concatenate(orgin_label)
    np.save(arr=orgin_label, file=os.path.join(STORE_PATH, test_area[0] + '_MC_origin.npy'))
    ensemble_label = np.concatenate(ensemble_label)
    np.save(arr=ensemble_label, file=os.path.join(STORE_PATH, test_area[0] + '_MC_ensemble.npy'))