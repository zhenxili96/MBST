#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:51:01 2018

@author: zhlixc
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import os
import os.path as osp

import numpy as np
import tensorflow as tf

import cv2
from cv2 import imwrite

from all_branch import convolutional_alexnet_arg_scope, embed_ori, embed_alexnet, \
                        embed_fn_0, embed_fn_1, embed_fn_2, embed_fn_3, embed_fn_4, \
                        embed_fn_5, embed_fn_6, embed_fn_7, embed_fn_8, embed_fn_9
from utils.infer_utils import get_exemplar_images
from utils.misc_utils import get_center


slim = tf.contrib.slim


class InferenceWrapper():
  """Model wrapper class for performing inference with a siamese model."""

  def __init__(self):
    self.image = None
    self.target_bbox_feed = None
    self.search_images = None
    self.embeds = None
    self.templates = None
    self.init = None
    self.response_up = None
    self.classid = None
    self.init_classid = None
    

  def build_graph_from_config(self, model_config, track_config, checkpoint_path):
    """Build the inference graph and return a restore function."""
    self.build_model()
    
    ema = tf.train.ExponentialMovingAverage(0)
    variables_to_restore = ema.variables_to_restore(moving_avg_variables=[])

    # Filter out State variables
    variables_to_restore_filterd = {}
    for key, value in variables_to_restore.items():
      if key.split('/')[1] != 'State':
              if "alex_branch" not in key:
                  if "vggf_branch" not in key:
                      variables_to_restore_filterd[key] = value
    
    saver = tf.train.Saver(variables_to_restore_filterd)
    

    if osp.isdir(checkpoint_path):
      #checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: {}".format(checkpoint_path))

    def _restore_fn(sess):
      logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path))
      logging.info("Restore CANet...")

    return _restore_fn

  def build_model(self):
    self.build_inputs()
    self.build_search_images()
    self.build_template()
    self.build_detection()
    self.build_upsample()

  def build_inputs(self):
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.to_float(image)
    self.image = image
    self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                           shape=[4],
                                           name='target_bbox_feed')  # center's y, x, height, width
    self.classid = tf.placeholder(tf.string, name="classid")

  def build_search_images(self):
    """Crop search images from the input image based on the last target position

    1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
    2. Crop an image patch as large as x_image_size centered at the target center.
    3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
    """

    size_z = 127
    size_x = 255
    context_amount = 0.5

    num_scales = 3
    scales = np.arange(num_scales) - get_center(num_scales)
    assert np.sum(scales) == 0, 'scales should be symmetric'
    search_factors = [1.0375 ** x for x in scales]

    frame_sz = tf.shape(self.image)
    target_yx = self.target_bbox_feed[0:2]
    target_size = self.target_bbox_feed[2:4]
    avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

    # Compute base values
    base_z_size = target_size
    base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size)
    base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
    base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
    d_search = (size_x - size_z) / 2.0
    base_pad = tf.div(d_search, base_scale_z)
    base_s_x = base_s_z + 2 * base_pad
    base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

    boxes = []
    for factor in search_factors:
      s_x = factor * base_s_x
      frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
      topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
      bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
      box = tf.concat([topleft, bottomright], axis=0)
      boxes.append(box)
    boxes = tf.stack(boxes)

    scale_xs = []
    for factor in search_factors:
      scale_x = base_scale_x / factor
      scale_xs.append(scale_x)
    self.scale_xs = tf.stack(scale_xs)

    # Note we use different padding values for each image
    # while the original implementation uses only the average value
    # of the first image for all images.
    image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
    image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                             box_ind=tf.zeros((3), tf.int32),
                                             crop_size=[size_x, size_x])
    self.search_images = image_cropped + avg_chan

  def get_image_embedding(self, images, classid, is_training=False):
    embed = tf.case(
            pred_fn_pairs=[
                    (tf.equal(classid, 'alx', name="eq11"), lambda : embed_alexnet(images)),
                    (tf.equal(classid, '0', name="eq0"), lambda : embed_fn_0(images, is_training)),
                    (tf.equal(classid, '1', name="eq1"), lambda : embed_fn_1(images, is_training)),
                    (tf.equal(classid, '2', name="eq2"), lambda : embed_fn_2(images, is_training)),
                    (tf.equal(classid, '3', name="eq3"), lambda : embed_fn_3(images, is_training)),
                    (tf.equal(classid, '4', name="eq4"), lambda : embed_fn_4(images, is_training)),
                    (tf.equal(classid, '5', name="eq5"), lambda : embed_fn_5(images, is_training)),
                    (tf.equal(classid, '6', name="eq6"), lambda : embed_fn_6(images, is_training)),
                    (tf.equal(classid, '7', name="eq7"), lambda : embed_fn_7(images, is_training)),
                    (tf.equal(classid, '8', name="eq8"), lambda : embed_fn_8(images, is_training)),
                    (tf.equal(classid, '9', name="eq9"), lambda : embed_fn_9(images, is_training)),
                    (tf.equal(classid, 'ori', name="eq10"), lambda : embed_ori(images, is_training))],
                    exclusive=False,
                    name="case1")
    return embed


  def build_template(self):

    # Exemplar image lies at the center of the search image in the first frame
    exemplar_images = get_exemplar_images(self.search_images, [127,127])
    templates = self.get_image_embedding(exemplar_images, self.classid)
    center_scale = int(get_center(3))
    center_template = tf.identity(templates[center_scale])
    templates = tf.stack([center_template for _ in range(3)])

    with tf.variable_scope('target_template'):
      # Store template in Variable such that we don't have to feed this template every time.
      with tf.variable_scope('State'):
        state = tf.get_variable('exemplar',
                                initializer=tf.zeros(templates.get_shape().as_list(), dtype=templates.dtype),
                                trainable=False)
        with tf.control_dependencies([templates]):
          self.init = tf.assign(state, templates, validate_shape=True)
        self.templates = state

  def build_detection(self):
    self.embeds = self.get_image_embedding(self.search_images, self.classid)
    with tf.variable_scope('detection'):
      def _translation_match(x, z):
        x = tf.expand_dims(x, 0)  # [batch, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, out_channels]
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

      output = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds, self.templates), dtype=self.embeds.dtype)  # of shape [16, 1, 17, 17, 1]
      output = tf.squeeze(output, [1, 4])  # of shape e.g. [16, 17, 17]

      bias = tf.get_variable('biases', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      response = 1e-3 * output + bias
      self.response = response
      #print ("response {}".format(self.response))

  def build_upsample(self):
    """Upsample response to obtain finer target position"""
    with tf.variable_scope('upsample'):
      response = tf.expand_dims(self.response, 3)
      up_method = 'bicubic'
      methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                 'bicubic': tf.image.ResizeMethod.BICUBIC}
      up_method = methods[up_method]
      response_spatial_size = self.response.get_shape().as_list()[1:3]
      up_size = [s * 16 for s in response_spatial_size]
      response_up = tf.image.resize_images(response,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up = tf.squeeze(response_up, [3])
      self.response_up = response_up
      #print ("response_up {}".format(self.response_up))

  def initialize(self, sess, input_feed, fixed_classid=None):
    image_path, target_bbox = input_feed
    roi_bbox = [[target_bbox[0]-target_bbox[2]/2, target_bbox[1]-target_bbox[3]/2, target_bbox[0]+target_bbox[2]/2, target_bbox[1]+target_bbox[3]/2]]
    init = sess.run([self.init], feed_dict={'filename:0': image_path,
                                      "target_bbox_feed:0": target_bbox,
                                      "classid:0":str(fixed_classid)})

  def inference_step(self, sess, input_feed, fixed_classid=None):
    image_path, target_bbox = input_feed

    image_cropped_op = self.search_images 
    image_cropped, scale_xs, response_output = sess.run(
      fetches=[image_cropped_op, self.scale_xs, self.response_up],
      feed_dict={
        "filename:0": image_path,
        "target_bbox_feed:0": target_bbox, 
        self.classid: str(fixed_classid), })

    output = {
      'image_cropped': image_cropped,
      'scale_xs': scale_xs,
      'response': response_output}
    return output, None