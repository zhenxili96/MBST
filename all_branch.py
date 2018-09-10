#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:16:42 2018

@author: zhlixc
"""

import tensorflow as tf
from utils.misc_utils import get
import logging
import numpy as np
import scipy.io as sio

import configuration

slim = tf.contrib.slim

AlexNet_Model_Path = "/home/travail/dev/GitRepo/SiamFC-TensorFlow/MulSiamFC/pretrained_model/bvlc_alexnet.npy"



def convolutional_alexnet_arg_scope(embed_config,
                                    trainable=True,
                                    is_training=False):
  """Defines the default arg scope.

  Args:
    embed_config: A dictionary which contains configurations for the embedding function.
    trainable: If the weights in the embedding function is trainable.
    is_training: If the embedding function is built for training.

  Returns:
    An `arg_scope` to use for the convolutional_alexnet models.
  """
  # Only consider the model to be in training mode if it's trainable.
  # This is vital for batch_norm since moving_mean and moving_variance
  # will get updated even if not trainable.
  is_model_training = trainable and is_training

  if get(embed_config, 'use_bn', True):
    batch_norm_scale = get(embed_config, 'bn_scale', True)
    batch_norm_decay = 1 - get(embed_config, 'bn_momentum', 3e-4)
    batch_norm_epsilon = get(embed_config, 'bn_epsilon', 1e-6)
    batch_norm_params = {
      "scale": batch_norm_scale,
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # Epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "trainable": trainable,
      "is_training": is_model_training,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
    normalizer_fn = slim.batch_norm
  else:
    batch_norm_params = {}
    normalizer_fn = None

  weight_decay = get(embed_config, 'weight_decay', 5e-4)
  if trainable:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  init_method = get(embed_config, 'init_method', 'kaiming_normal')
  if is_model_training:
    logging.info('embedding init method -- {}'.format(init_method))
  if init_method == 'kaiming_normal':
    # The same setting as siamese-fc
    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,
      padding='VALID',
      trainable=trainable,
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.batch_norm], is_training=is_model_training) as arg_sc:
        return arg_sc
    

def embed_fn_core(images, is_training=True):
    model_config = configuration.MODEL_CONFIG
    config = model_config['embed_config']
    arg_scope = convolutional_alexnet_arg_scope(config,
                                                trainable=config['train_embedding'],
                                                is_training=is_training)
    with slim.arg_scope(arg_scope):
        scope='convolutional_alexnet'
        with tf.variable_scope(scope, 'convolutional_alexnet', [images], reuse=tf.AUTO_REUSE) as sc:
          end_points_collection = sc.name + '_end_points'
          with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
              net = images
              net_c1 = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')
              net_p1 = slim.max_pool2d(net_c1, [3, 3], 2, scope='pool1')
              with tf.variable_scope('conv2'):
                b1, b2 = tf.split(net_p1, 2, 3)
                b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
                # The original implementation has bias terms for all convolution, but
                # it actually isn't necessary if the convolution layer is followed by a batch
                # normalization layer since batch norm will subtract the mean.
                b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
                net_c2 = tf.concat([b1, b2], 3)
              net_p2 = slim.max_pool2d(net_c2, [3, 3], 2, scope='pool2')
              net_c3 = slim.conv2d(net_p2, 384, [3, 3], 1, scope='conv3')
              with tf.variable_scope('conv4'):
                b1, b2 = tf.split(net_c3, 2, 3)
                b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
                b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
                net_c4 = tf.concat([b1, b2], 3)
              # Conv 5 with only convolution, has bias
              with tf.variable_scope('conv5'):
                with slim.arg_scope([slim.conv2d],
                                    activation_fn=None, normalizer_fn=None):
                  b1, b2 = tf.split(net_c4, 2, 3)
                  b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
                  b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
                net_c5 = tf.concat([b1, b2], 3)
              return net_c5
    
def embed_fn_0(images, is_training=True):
    with tf.variable_scope("fn0", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)

def embed_fn_1(images, is_training=True):
    with tf.variable_scope("fn1", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)

def embed_fn_2(images, is_training=True):
    with tf.variable_scope("fn2", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)
                             
def embed_fn_3(images, is_training=True):
    with tf.variable_scope("fn3", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)

def embed_fn_4(images, is_training=True):
    with tf.variable_scope("fn4", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)
    
def embed_fn_5(images, is_training=True):
    with tf.variable_scope("fn5", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)
                             
def embed_fn_6(images, is_training=True):
    with tf.variable_scope("fn6", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)

def embed_fn_7(images, is_training=True):
    with tf.variable_scope("fn7", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)
    
def embed_fn_8(images, is_training=True):
    with tf.variable_scope("fn8", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)
                             
def embed_fn_9(images, is_training=True):
    with tf.variable_scope("fn9", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)
    
def embed_ori(images, is_training=False):
    with tf.variable_scope("ori", reuse=tf.AUTO_REUSE):
        return embed_fn_core(images, is_training)
    
def get_pretrained_alexnet():
    all_params = np.load(AlexNet_Model_Path).item()
    return all_params
    
def embed_alexnet(images):
## refer to following links and SA-Net
# https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
# https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward.py
    with tf.variable_scope("alex_branch", reuse=tf.AUTO_REUSE):
        net_data = get_pretrained_alexnet()
        def conv(data, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
            '''From https://github.com/ethereon/caffe-tensorflow
            '''
            c_i = data.get_shape()[-1]
            assert c_i%group==0
            assert c_o%group==0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
            
            
            if group==1:
                conv = convolve(data, kernel)
            else:
                input_groups = tf.split(data, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)
            return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
            
    
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 2; s_w = 2
        conv1W = tf.Variable(net_data["conv1"][0], trainable=False, name="conv1W")
        conv1b = tf.Variable(net_data["conv1"][1], trainable=False, name="conv1b")
        conv1_in = conv(images, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1)
        conv1 = tf.nn.relu(conv1_in)
        
        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        
        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(net_data["conv2"][0], trainable=False, name="conv2W")
        conv2b = tf.Variable(net_data["conv2"][1], trainable=False, name="conv2b")
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv2 = tf.nn.relu(conv2_in)
        
        
        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        
        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(net_data["conv3"][0], trainable=False, name="conv3W")
        conv3b = tf.Variable(net_data["conv3"][1], trainable=False, name="conv3b")
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv3 = tf.nn.relu(conv3_in)
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(net_data["conv4"][0], trainable=False, name="conv4W")
        conv4b = tf.Variable(net_data["conv4"][1], trainable=False, name="conv4b")
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv4 = tf.nn.relu(conv4_in)
        
        
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(net_data["conv5"][0], trainable=False, name="conv5W")
        conv5b = tf.Variable(net_data["conv5"][1], trainable=False, name="conv5b")
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv5 = tf.nn.relu(conv5_in)
        return conv5
    
