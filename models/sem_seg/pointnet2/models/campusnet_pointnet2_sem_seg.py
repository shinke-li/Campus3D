import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

RADIUS = [0.1, 0.2, 0.4, 0.8]
DATA_DIM = 3


def placeholder_inputs(batch_size, num_point, num_heads):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, DATA_DIM))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point, num_heads))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_heads))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    if isinstance(num_class, int):
        num_class = [num_class]
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    if DATA_DIM > 3:
        l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
        l0_points = tf.slice(point_cloud, [0, 0, 0], [-1, -1, DATA_DIM - 3])
    else:
        l0_xyz = point_cloud
        l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Decoders layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=RADIUS[0], nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=RADIUS[1], nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=RADIUS[2], nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=RADIUS[3], nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net

    nets = []
    for i, n_cls in enumerate(num_class):
        new_net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='head{}_dp1'.format(i))
        new_net = tf_util.conv1d(new_net,  n_cls, 1, padding='VALID', activation_fn=None, scope='head{}_fc2'.format(i))
        nets.append(new_net)
    return nets, end_points


def get_model_decoder(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    if isinstance(num_class, int):
        num_class = [num_class]
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    if DATA_DIM > 3:
        l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
        l0_points = tf.slice(point_cloud, [0, 0, 0], [-1, -1, DATA_DIM - 3])
    else:
        l0_xyz = point_cloud
        l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Decoders layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=RADIUS[0], nsample=32, mlp=[32,32,64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=RADIUS[1], nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=RADIUS[2], nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=RADIUS[3], nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    nets = []
    for i, n_cls in enumerate(num_class):
        # Feature Propagation layers
        l3_points_new = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer{}_1'.format(i))
        l2_points_new = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_new, [256,256], is_training, bn_decay, scope='fa_layer{}_2'.format(i))
        l1_points_new = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points_new, [256,128], is_training, bn_decay, scope='fa_layer{}_3'.format(i))
        l0_points_new = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_new, [128,128,128], is_training, bn_decay, scope='fa_layer{}_4'.format(i))

        # FC layers
        net = tf_util.conv1d(l0_points_new, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='head{}_fc1'.format(i), bn_decay=bn_decay)
        end_points['feats{}'.format(i)] = net
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='head{}_dp1'.format(i))
        net = tf_util.conv1d(net,  n_cls, 1, padding='VALID', activation_fn=None, scope='head{}_fc2'.format(i))
        nets.append(net)

    return nets, end_points


def get_loss(preds, labels, smpws, loss_weights, num_heads):
    """ pred: [BxNxC1, ..., BxNxCn] (n = 1,..., NUM_HEADS)
        label: BxNxNUM_HEADS,
	    smpw: BxNxNUM_HEADS
	"""

    for i in range(num_heads):
        loss = loss_weights[i] * tf.losses.sparse_softmax_cross_entropy(
            labels=labels[..., i],
            logits=preds[i],
            weights=smpws[..., i])
        tf.summary.scalar('loss head{}'.format(i), loss)
        tf.add_to_collection('losses', loss)
    return


def get_L2_consistency_loss(matrices, preds, weights):
    probs = [tf.nn.softmax(pred) for pred in preds]
    for i, matrix in enumerate(matrices):
        m = tf.constant(matrix, dtype=tf.float32)
        prob_ = tf.tensordot(m, probs[i + 1], axes=((1,), (2,)))
        prob_ = tf.transpose(prob_, [1, 2, 0])
        consist_loss = weights[i] * tf.square(tf.nn.l2_loss(prob_ - probs[i]))
        tf.add_to_collection('consistency_losses', consist_loss)
    return


def get_hinge_consistency_loss(matrices , preds, weights):
    gather_id = [np.argmax(m, axis=0)for m in matrices]
    probs = [tf.nn.softmax(pred) for pred in preds]
    for i, gid in enumerate(gather_id):
        id = tf.constant(gid, dtype=tf.int64)
        prob_ = tf.gather(probs[i], indices=id, axis=-1)
        subtract_prob = tf.subtract(probs[i + 1], prob_)
        consist_loss = weights[i] * tf.square(tf.nn.l2_loss(tf.nn.relu(subtract_prob)))
        tf.add_to_collection('consistency_losses', consist_loss)
    return


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
