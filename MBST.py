#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:50:41 2018

@author: zhlixc
"""

import tensorflow as tf
import numpy as np


from datasets.dataloader import DataLoader


from all_branch import convolutional_alexnet_arg_scope, embed_fn_0, \
 embed_fn_1, embed_fn_2, embed_fn_3, embed_fn_4, embed_fn_5, embed_fn_6, \
 embed_fn_7, embed_fn_8, embed_fn_9, embed_ori
from utils.train_utils import construct_gt_score_maps
from metrics.track_metrics import center_dist_error, center_score_error



class MBST:
    
    def __init__(self, configuration):
        self.model_config = configuration.MODEL_CONFIG
        self.train_config = configuration.TRAIN_CONFIG
        self.data_config = self.train_config['train_data_config']
        self.mode = "train"
        
        
    def build_inputs(self):
        self.dataloader = DataLoader(self.data_config, True)
        self.dataloader.build()
        exemplars, instances, clusters = self.dataloader.get_one_batch()
        self.exemplars = tf.to_float(exemplars)
        self.instances = tf.to_float(instances)
        self.classid = clusters[0]
        
    def build_embedding(self):

        self.templates = tf.case(
                pred_fn_pairs=[
                        (tf.equal(self.classid, '0', name="eq0"), lambda : embed_fn_0(self.exemplars)),
                        (tf.equal(self.classid, '1', name="eq1"), lambda : embed_fn_1(self.exemplars)),
                        (tf.equal(self.classid, '2', name="eq2"), lambda : embed_fn_2(self.exemplars)),
                        (tf.equal(self.classid, '3', name="eq3"), lambda : embed_fn_3(self.exemplars)),
                        (tf.equal(self.classid, '4', name="eq4"), lambda : embed_fn_4(self.exemplars)),
                        (tf.equal(self.classid, '5', name="eq5"), lambda : embed_fn_5(self.exemplars)),
                        (tf.equal(self.classid, '6', name="eq6"), lambda : embed_fn_6(self.exemplars)),
                        (tf.equal(self.classid, '7', name="eq7"), lambda : embed_fn_7(self.exemplars)),
                        (tf.equal(self.classid, '8', name="eq8"), lambda : embed_fn_8(self.exemplars)),
                        (tf.equal(self.classid, '9', name="eq9"), lambda : embed_fn_9(self.exemplars)),
                        (tf.equal(self.classid, 'ori', name="eq_ori"), lambda : embed_ori(self.exemplars))],
                        exclusive=False,
                        name="case1")
        self.instance_embeds = tf.case(
                pred_fn_pairs=[
                        (tf.equal(self.classid, '0', name="eq0"), lambda : embed_fn_0(self.instances)),
                        (tf.equal(self.classid, '1', name="eq1"), lambda : embed_fn_1(self.instances)),
                        (tf.equal(self.classid, '2', name="eq2"), lambda : embed_fn_2(self.instances)),
                        (tf.equal(self.classid, '3', name="eq3"), lambda : embed_fn_3(self.instances)),
                        (tf.equal(self.classid, '4', name="eq4"), lambda : embed_fn_4(self.instances)),
                        (tf.equal(self.classid, '5', name="eq5"), lambda : embed_fn_5(self.instances)),
                        (tf.equal(self.classid, '6', name="eq6"), lambda : embed_fn_6(self.instances)),
                        (tf.equal(self.classid, '7', name="eq7"), lambda : embed_fn_7(self.instances)),
                        (tf.equal(self.classid, '8', name="eq8"), lambda : embed_fn_8(self.instances)),
                        (tf.equal(self.classid, '9', name="eq9"), lambda : embed_fn_9(self.instances)),
                        (tf.equal(self.classid, 'ori', name="eq_ori"), lambda : embed_ori(self.instances))],
                        exclusive=False,
                        name="case1")

    def build_detection(self, reuse=False):
        with tf.variable_scope('detection', reuse=reuse):
            def _translation_match(x, z):  # translation match for one example within a batch
                x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
                z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
                return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

            output = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                         (self.instance_embeds, self.templates),
                         dtype=self.instance_embeds.dtype)
            output = tf.squeeze(output, [1, 4])  # of shape e.g., [8, 15, 15]

            # Adjust score, this is required to make training possible.
            config = self.model_config['adjust_response_config']
            self.bias = tf.get_variable('biases', [1],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                         trainable=config['train_bias'])
            response = config['scale'] * output + self.bias
            self.response = response

    def build_loss(self):
        response = self.response
        response_size = response.get_shape().as_list()[1:3]  # [height, width]

        self.gt = construct_gt_score_maps(response_size,
                                 self.data_config['batch_size'],
                                 self.model_config['embed_config']['stride'],
                                 self.train_config['gt_config'])

        with tf.name_scope('Loss'):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=response,
                                                     labels=self.gt)
            with tf.name_scope('Balance_weights'):
                n_pos = tf.reduce_sum(tf.to_float(tf.equal(self.gt[0], 1)))
                n_neg = tf.reduce_sum(tf.to_float(tf.equal(self.gt[0], 0)))
                w_pos = 0.5 / n_pos
                w_neg = 0.5 / n_neg
                class_weights = tf.where(tf.equal(self.gt, 1),
                                         w_pos * tf.ones_like(self.gt),
                                         tf.ones_like(self.gt))
                class_weights = tf.where(tf.equal(self.gt, 0),
                                         w_neg * tf.ones_like(self.gt),
                                         class_weights)
                loss = loss * class_weights

            # Note that we use reduce_sum instead of reduce_mean since the loss has
            # already been normalized by class_weights in spatial dimension.
            loss = tf.reduce_sum(loss, [1, 2])
            
            batch_loss = tf.reduce_mean(loss, name='batch_loss')
            tf.losses.add_loss(batch_loss)
        
            total_loss = tf.losses.get_total_loss()
            self.batch_loss = batch_loss
            self.total_loss = total_loss

    def setup_global_step(self):
        global_step = tf.Variable(
                initial_value=0,
                name='global_step',
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self, reuse=False):
        """Creates all ops for training and evaluation"""
        with tf.name_scope(self.mode):
            self.build_inputs()
            self.build_embedding()
            self.build_detection(reuse=reuse)
            self.build_loss()
            self.setup_global_step()
 
