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


def train(cfg, train_data_loader, validation_data_loader, metric_module, logger, **kwargs):

    cfg = cfg
    METRIC_MODULE = metric_module

    BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    ENABLE_CONSISTENCY_LOSS = cfg.TRAIN.CONSISTENCY_LOSS
    HM = kwargs['h_matrices']
    CONSISTENCY_MATRIX = [HM[i+1, i] for i in range(len(HM.classes_num) - 1)]
    CONSISTENCY_LOSS_WEIGHTS = cfg.TRAIN.CONSISTENCY_WEIGHTS
    NUM_POINT = cfg.DATASET.SAMPLE.SETTING.NUM_POINTS_PER_SAMPLE
    MAX_EPOCH = cfg.TRAIN.MAX_EPOCH
    BASE_LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    OPTIMIZER = 'adam'
    DECAY_STEP = 200000
    DECAY_RATE = 0.7
    MODEL_NAME = 'campusnet_pointnet2_sem_seg'

    MODEL = importlib.import_module(MODEL_NAME)  # import network module
    MODEL_FILE = os.path.join(ROOT_DIR, 'models', MODEL_NAME + '.py')

    LOG_DIR = cfg.TRAIN.OUTPUT_DIR
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
    os.system('cp %s %s' % (os.path.abspath(__file__), LOG_DIR))  # bkp of train procedure
    LOG_FOUT = logger
    LOG_FOUT.info(str(cfg) + '\n')

    MODEL.DATA_DIM = 6
    MODEL.RADIUS = cfg.TRAIN.SETTING.RADIUS
    MODEL_PATH = cfg.TRAIN.PRETRAINED_MODEL_PATH
    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99

    HOSTNAME = socket.gethostname()


    def log_string(out_str):
        LOG_FOUT.info(out_str)
        #LOG_FOUT.flush()
        #print(out_str)

    def get_learning_rate(batch):
        learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            DECAY_STEP,  # Decay step.
            DECAY_RATE,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
        return learning_rate

    def get_bn_decay(batch):
        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch * BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def train_one_epoch(sess, ops, train_writer, data_loader):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        log_string(str(datetime.now()))

        cfs_mtx_list = [METRIC_MODULE.IouMetric(list(range(l))) for l in cfg.DATASET.DATA.LABEL_NUMBER]
        loss_sum = 0.0
        consist_loss_sum = 0.0

        for batch_idx, data_ in enumerate(data_loader):
            points_centered, labels, colors, label_weights = data_
            if labels.shape[0] < cfg.TRAIN.BATCH_SIZE:
                break
            points_clrs = np.concatenate([points_centered, colors], axis=-1)
            feed_dict = {ops['pointclouds_pl']: points_clrs,
                         ops['labels_pl']: labels,
                         ops['smpws_pl']: label_weights,
                         ops['is_training_pl']: is_training, }
            if ENABLE_CONSISTENCY_LOSS:
                summary, step, _, loss_val, pred_vals, consist_loss_val = sess.run([ops['merged'], ops['step'],
                                                                  ops['train_op'], ops['total_loss'], ops['preds'],ops['consistency_loss']],
                                                                 feed_dict=feed_dict)
                consist_loss_sum += consist_loss_val
            else:
                summary, step, _, loss_val, pred_vals = sess.run([ops['merged'], ops['step'],
                                                              ops['train_op'], ops['total_loss'], ops['preds']],
                                                             feed_dict=feed_dict)

            train_writer.add_summary(summary, step)
            loss_sum += loss_val

            for head_idx, pred_val in enumerate(pred_vals):
                pred_val = np.argmax(pred_val, 2)
                cfs_mtx_list[head_idx].update(pred_val, labels[..., head_idx])

            if (batch_idx + 1) % 200 == 0:#%200 == 0:
                log_string(' ---- batch: %03d ----' % (batch_idx + 1))
                log_string('mean loss: %f' % (loss_sum / 200))
                if ENABLE_CONSISTENCY_LOSS:
                    log_string('mean consist loss: %f' % (consist_loss_sum / 200))
                log_string('mean IoU: {}'.format({i: mtx.avg_iou() for i, mtx in enumerate(cfs_mtx_list)}))
                cfs_mtx_list = [METRIC_MODULE.IouMetric(list(range(l))) for l in cfg.DATASET.DATA.LABEL_NUMBER]
                loss_sum = 0
                consist_loss_sum = 0
        print('Final {}'.format(batch_idx))

    def eval_one_epoch(sess, ops, data_loader, epoch_num):
        """ ops: dict mapping from string to tf ops """
        is_training = False
        log_string(str(datetime.now()))
        log_string('---- EPOCH %03d EVALUATION ----' % (epoch_num))
        cfs_mtx_list = [METRIC_MODULE.IouMetric(list(range(l))) for l in cfg.DATASET.DATA.LABEL_NUMBER]
        all_heads_label = [[] for _ in range(len(HM.classes_num))]
        for batch_idx, data_ in enumerate(data_loader):
            points_centered, labels, colors, label_weights = data_
            init_ind, point_clrs, labels, label_weights = full_batch_size(
                data_loader.batch_size,
                np.concatenate([points_centered, colors], axis=-1),
                labels,
                label_weights)

            feed_dict = {ops['pointclouds_pl']: point_clrs,
                         ops['labels_pl']: labels,
                         ops['is_training_pl']: is_training, }
            pred_vals = sess.run([ops['preds']], feed_dict=feed_dict)

            for head_idx, pred_val in enumerate(pred_vals[0]):
                pred_val = np.argmax(pred_val[init_ind], 2)
                cfs_mtx_list[head_idx].update(pred_val, labels[init_ind, ..., head_idx])
                all_heads_label[head_idx].append(pred_val.reshape(-1))
        all_heads_label = np.asarray([np.concatenate(l) for l in all_heads_label]).transpose()
        #print(all_heads_label.shape)
        scores = METRIC_MODULE.HierarchicalConsistency.cal_consistency_rate(HM, all_heads_label)

            #direct_labels = [np.argmax(lgs, axis=1) for lgs in logits]
            # print([l.shape for l in direct_labels])
            # print([l.shape for l in direct_labels])
            #scores =
        log_string('consistency score: {}'.format(scores))
        log_string('eval IoU: {}'.format('\n'.join([str(m.iou()) for m in cfs_mtx_list])))
        log_string('eval avg class IoU: {}'.format('\n'.join([str(m.avg_iou()) for m in cfs_mtx_list])))


    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, len(cfg.DATASET.DATA.LABEL_NUMBER))
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            preds, end_points = MODEL.get_model_decoder(pointclouds_pl, is_training_pl, cfg.DATASET.DATA.LABEL_NUMBER, bn_decay=bn_decay)
            total_loss = MODEL.get_loss(preds, labels_pl, smpws_pl, cfg.TRAIN.LOSS_WEIGHTS, len(cfg.DATASET.DATA.LABEL_NUMBER))

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_seg_loss')
            if ENABLE_CONSISTENCY_LOSS:
                consistency_loss = MODEL.get_hinge_consistency_loss(CONSISTENCY_MATRIX, preds, CONSISTENCY_LOSS_WEIGHTS)
                consist_losses = tf.get_collection('consistency_losses')
                total_consist_loss = tf.add_n(consist_losses, name='consistency_loss')
                total_loss = total_loss + total_consist_loss
            tf.summary.scalar('total_loss', total_loss)
            #for l in losses + [total_loss]:
                #tf.summary.scalar(l.op.name, l)

            for ind, pred in enumerate(preds):
                correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl[..., ind]))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                tf.summary.scalar('accuracy_{}head'.format(ind), accuracy)

            #print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=100)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False




        sess = tf.Session(config=config)
        ckptstate = tf.train.get_checkpoint_state(MODEL_PATH)
        log_string('Pretrained state:{}'.format(ckptstate))
        if MODEL_PATH:
            saver.restore(sess, MODEL_PATH)
            log_string('Restore model: {}'.format(MODEL_PATH))

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if not MODEL_PATH:
            init = tf.global_variables_initializer()
            sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'preds': preds,
               'total_loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}
        if ENABLE_CONSISTENCY_LOSS:
            ops.update({'consistency_loss': total_consist_loss})
        try:
            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                train_one_epoch(sess, ops, train_writer, train_data_loader)
                eval_one_epoch(sess, ops, validation_data_loader,epoch)
                # Save the variables to disk.
                if epoch % 3 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "epoch{}_model.ckpt".format(epoch)))
                    log_string("Model saved in file: %s" % save_path)
        except Exception as e:
            traceback.print_exc()
        #LOG_FOUT.close()





        



if __name__ == "__main__":
    pass
    #log_string('pid: %s'%(str(os.getpid())))
    #train()
    #LOG_FOUT.close()
