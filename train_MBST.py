#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:43:47 2018

@author: zhlixc
"""


import tensorflow as tf
import numpy as np
import os
import random

import configuration
from MBST import MBST
from utils.misc_utils import auto_select_gpu, mkdir_p, save_cfgs


model_config = configuration.MODEL_CONFIG
train_config = configuration.TRAIN_CONFIG
track_config = configuration.TRACK_CONFIG


os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
check_dir = "/home/travail/dev/GitRepo/SiamFC-TensorFlow/Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained"
mode_save_path = "/home/travail/dev/GitRepo/SiamFC-TensorFlow/MulSiamFC/Logs/checkpoints/on_pretrained_plus_oribranch/model.ckpt"
log_dir = "/home/travail/dev/GitRepo/SiamFC-TensorFlow/MulSiamFC/Logs"

g = tf.Graph()
with g.as_default():
    # Set fixed seed for reproducible experiments
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])
    
    mbst = MBST(configuration)
    mbst.build()
    
    train_step = tf.train.GradientDescentOptimizer(1e-6).minimize(mbst.total_loss)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    # Dynamically allocate GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session()
    checkpoint_path = tf.train.latest_checkpoint(check_dir)

    

    sess.run(init_local)
    sess.run(init_global)
    ## load pre-trained Siamfc
    load_pretrained_siamfc = True
    if load_pretrained_siamfc:
        from restore import get_restore_list
        
        fn_scope = ["fn0", "fn1", "fn2", "fn3", "fn4", 
                    "fn5", "fn6", "fn7", "fn8", "fn9", "ori"]
        for scope_item in fn_scope:
            restore_list = get_restore_list(scope_item)
            saver = tf.train.Saver(var_list=restore_list)
            saver.restore(sess, checkpoint_path)
        restore_list["detection/biases"] = mbst.bias
        saver = tf.train.Saver(var_list=restore_list)
        saver.restore(sess, checkpoint_path)
        print("load_pretrained_siamfc: {}".format(checkpoint_path))
    
    
 

    summarizer = tf.summary.FileWriter(log_dir, sess.graph)
    BL = [9.99,9.99,9.99,9.99,9.99,9.99,9.99,9.99,9.99,9.99]
    sum_bl = tf.Summary()
    sum_bl.value.add(tag="class_0_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_1_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_2_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_3_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_4_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_5_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_6_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_7_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_8_loss", simple_value=9.99)
    sum_bl.value.add(tag="class_9_loss", simple_value=9.99)
    
    
    
    model_saver = tf.train.Saver()
    save_path = model_saver.save(sess, mode_save_path, global_step=0)
    print("Model saved in path: %s" % save_path)
    #g.finalize()
    for i in range(999999):
        _, batch_loss, batch_class, exemplars, instances = sess.run([
                train_step, mbst.batch_loss, mbst.classid,
                mbst.exemplars, mbst.instances])
        total_loss = mbst.total_loss
        if (i%10 == 0):
            BL[int(batch_class)] = batch_loss
            print ("step: %06d class:%s BL: [0]%.2f [1]%.2f [2]%.2f [3]%.2f [4]%.2f [5]%.2f [6]%.2f [7]%.2f [8]%.2f [9]%.2f" 
                   % (i, batch_class, BL[0], BL[1], BL[2], BL[3], BL[4], BL[5], BL[6], BL[7], BL[8], BL[9]))
            for j in range(10):
                sum_bl.value[j].simple_value = BL[j]
            summarizer.add_summary(sum_bl, i)
            summarizer.flush()
            
        if i>0 and i%1000==0:
            save_path = model_saver.save(sess, mode_save_path, global_step=i)
            print("Model saved in path: %s" % save_path)
        if i>30000:
            break
    summarizer.close()
