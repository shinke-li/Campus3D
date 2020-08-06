'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import traceback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
#provider = importlib.import_module('utils.provider')
#tf_util = importlib.import_module('utils.tf_util')
import provider
import tf_util
#USE NORMAL set true before

def full_batch_size(batch_size, *np_args):
    sample_size = np_args[0].shape[0]
    init_ind = np.arange(sample_size)
    if sample_size < batch_size:
        res_ind = np.random.randint(0, sample_size, (batch_size - sample_size, ))
        np_args = [np.concatenate([arr, arr[res_ind]]) for arr in np_args]

    return tuple([init_ind] + list(np_args))

def find_best_model(num):
    return "epoch{}_model.ckpt".format(num)

def test(cfg, test_loader, log, ckpt_name):

    cfg = cfg

    BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    NUM_POINT = cfg.DATASET.SAMPLE.SETTING.NUM_POINTS_PER_SAMPLE
    MODEL_NAME = 'campusnet_pointnet2_sem_seg'

    MODEL = importlib.import_module(MODEL_NAME)  # import network module

    LOG_DIR = cfg.TRAIN.OUTPUT_DIR
    MODEL_PATH = os.path.join(LOG_DIR, ckpt_name)#find_best_model(108))
    log.info('Load model {}'.format(MODEL_PATH))

    MODEL.DATA_DIM = 6
    MODEL.RADIUS = cfg.TRAIN.SETTING.RADIUS


    HOSTNAME = socket.gethostname()

    log.debug('Do test')

    def eval_one_epoch(sess, ops,  data_loader):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        logits_collections = [[] for _ in range(len(cfg.DATASET.DATA.LABEL_NUMBER))]
        points_collections = []
        for batch_idx, data_ in enumerate(data_loader):
            if batch_idx % 100 == 0:
                log.debug('{} batches finished'.format(batch_idx))
            points_centered, labels, colors, raw_points = data_
            init_ind, point_clrs, labels = full_batch_size(
                data_loader.batch_size,
                np.concatenate([points_centered, colors], axis=-1),
                labels,
                )

            feed_dict = {ops['pointclouds_pl']: point_clrs,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training, }
            pred_vals = sess.run([ops['preds']], feed_dict=feed_dict)[0]
            points_collections.append(raw_points)
            for pred, collects in zip(pred_vals, logits_collections):
                collects.append(pred[init_ind])
        return np.concatenate(points_collections), [np.concatenate(lgs) for lgs in logits_collections]



    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, len(cfg.DATASET.DATA.LABEL_NUMBER))
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Get model and loss
            preds, end_points = MODEL.get_model_decoder(pointclouds_pl, is_training_pl, cfg.DATASET.DATA.LABEL_NUMBER)

            #for l in losses + [total_loss]:
                #tf.summary.scalar(l.op.name, l)

            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        saver.restore(sess, MODEL_PATH)


        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'preds': preds,
               'step': batch,
               'end_points': end_points}
        points, logits = eval_one_epoch(sess, ops,  test_loader)
        points = points.reshape(points.shape[0] * points.shape[1], 3)
        logits = [logit.reshape(logit.shape[0] * logit.shape[1], logit.shape[2]) for logit in logits]

        return points, logits







        



if __name__ == "__main__":
    pass
    #log_string('pid: %s'%(str(os.getpid())))
    #train()
    #LOG_FOUT.close()
