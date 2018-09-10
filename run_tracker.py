#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:05:40 2018

@author: zhlixc
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import os.path as osp
import sys
from glob import glob

import tensorflow as tf
from sacred import Experiment

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

import mbst_inference_wrapper
from mbst_tracker import Tracker
from utils.infer_utils import Rectangle
from utils.misc_utils import auto_select_gpu, mkdir_p, sort_nicely, load_cfgs

ex = Experiment()

      
      
def readbbox(file):
  with open(file, 'r') as f:
    lines = f.readlines()
    bboxs = [[float(val) for val in line.strip().replace(' ', ',').replace('\t', ',').split(',')] for line in lines]
  return bboxs
      
def cal_IOU(pred_bboxs, gt_bboxs):
      if len(pred_bboxs) != len(gt_bboxs):
          print ("ERROR IN CAL IOU {0} {1}".format(len(pred_bboxs), len(gt_bboxs)))
      sum_iou = 0
      for i, item in enumerate(pred_bboxs):
          xA = max(item[0], gt_bboxs[i][0])
          yA = max(item[1], gt_bboxs[i][1])
          xB = min(item[0]+item[2], gt_bboxs[i][0]+gt_bboxs[i][2])
          yB = min(item[1]+item[3], gt_bboxs[i][1]+gt_bboxs[i][3])
          
          interArea = max(0, xB-xA+1)*max(0, yB-yA+1)
          boxAArea = item[2]*item[3]
          boxBArea = gt_bboxs[i][2]*gt_bboxs[i][3]
          
          iou = interArea / float(boxAArea+boxBArea-interArea)
          sum_iou = sum_iou + iou
      return sum_iou/len(pred_bboxs)
          
      

checkpoint = "/home/travail/dev/GitRepo/SiamFC-TensorFlow/MulSiamFC/Logs/checkpoints/on_pretrained_plus_oribranch/model.ckpt-30000"
video_dirs = ['/home/travail/dev/GitRepo/SiamFC-TensorFlow/assets/Bolt']

# =============================================================================
# 
# @ex.automain
# def main(checkpoint, input_files):
# =============================================================================
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

model_config, _, track_config = load_cfgs(checkpoint)
track_config['log_level'] = 1

g = tf.Graph()
with g.as_default():
    model = mbst_inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint)


if not osp.isdir(track_config['log_dir']):
    logging.info('Creating inference directory: %s', track_config['log_dir'])
    mkdir_p(track_config['log_dir'])
  
    

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)


    
ALL_ResList = []

with tf.Session(graph=g, config=sess_config) as sess:
    
    ## used for initializing alexnet parameters
    init_global = tf.global_variables_initializer()
    sess.run(init_global)
    
    ## global initalizer must be run before restore
    restore_fn(sess)
    #print(sess.run(tf.report_uninitialized_variables()))
    

    av1 = tf.all_variables()
    tracker = Tracker(model, model_config=model_config, track_config=track_config)

    for video_dir in video_dirs:
      if not osp.isdir(video_dir):
        logging.warning('{} is not a directory, skipping...'.format(video_dir))
        continue

      video_name = osp.basename(video_dir)
      video_log_dir = "tmp"
      mkdir_p(video_log_dir)

      filenames = sort_nicely(glob(video_dir + '/img/*.jpg'))
      first_line = open(video_dir + '/groundtruth_rect.txt').readline()
      bb = [int(v) for v in first_line.strip().replace(' ', ',').replace('\t', ',').split(',')]
      init_bb = Rectangle(bb[0] - 1, bb[1] - 1, bb[2], bb[3])  # 0-index in python
      
      print("######{0},{1}".format(video_dir, len(filenames)))
# =============================================================================
#       for i in range(10):
#           print ("fixed classid: {}".format(i))
#           trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, str(i))
#           with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#             for region in trajectory:
#               rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                                 region.width, region.height)
#               f.write(rect_str)
#       
#           gt_bboxs = readbbox(osp.join(input_files, 'groundtruth_rect.txt'))
#           pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#           print ("MulSiamFC class  --- {0}  IOU --- {1}".format(str(i), cal_IOU(pred_bboxs, gt_bboxs)))
# =============================================================================
      
      #### origin branch
      print ("origin branch")
#      trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, 'alx')
      with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
        for region in trajectory:
          rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
                                            region.width, region.height)
          f.write(rect_str)
  
      gt_bboxs = readbbox(osp.join(video_dir, 'groundtruth_rect.txt'))
      pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
      print ("MulSiamFC class  ---  IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
      
# =============================================================================
#       #### alexnet branch
#       print ("alexnet branch")
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, "alx")
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)
#       gt_bboxs = readbbox(osp.join(video_dir, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---  IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
# =============================================================================
      
# =============================================================================
#       #### vggf branch
#       print ("vggf branch")
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, "vggf")
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)
#       gt_bboxs = readbbox(osp.join(video_dir, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---  IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
# =============================================================================
      
# =============================================================================
#       T = 10
#       weightList = [10.5, 6.2]
#       print ("dynamic branch t={0} weight={1}".format(T, weightList))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T, weightList)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)      
#       gt_bboxs = readbbox(osp.join(video_dir, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---  IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
# =============================================================================
      
# =============================================================================
#       T = 10
#       weightList = [10.0, 6.5]
#       print ("dynamic branch t={0} weight={1}".format(T, weightList))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T, weightList)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)      
#       gt_bboxs = readbbox(osp.join(video_dir, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---  IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
#       
#       T = 10
#       weightList = [9.5, 6.5]
#       print ("dynamic branch t={0} weight={1}".format(T, weightList))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T, weightList)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)      
#       gt_bboxs = readbbox(osp.join(video_dir, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---  IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
#       
#       T = 10
#       weightList = [9.5, 6]
#       print ("dynamic branch t={0} weight={1}".format(T, weightList))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T, weightList)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)      
#       gt_bboxs = readbbox(osp.join(video_dir, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---  IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
# =============================================================================
      
# =============================================================================
#       
#       #### dynamic branch
#       T = 1
#       print ("dynamic branch t={}".format(T))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T)
#       #trajectory = tracker.track(sess, init_bb, filenames, video_log_dir)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)
#   
#       gt_bboxs = readbbox(osp.join(input_files, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---   IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
#       
#       T = 5
#       print ("dynamic branch t={}".format(T))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T)
#       #trajectory = tracker.track(sess, init_bb, filenames, video_log_dir)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)
#   
#       gt_bboxs = readbbox(osp.join(input_files, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---   IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
#       
#       T = 10
#       print ("dynamic branch t={}".format(T))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T)
#       #trajectory = tracker.track(sess, init_bb, filenames, video_log_dir)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)
#   
#       gt_bboxs = readbbox(osp.join(input_files, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---   IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
#       
#       T = 15
#       print ("dynamic branch t={}".format(T))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T)
#       #trajectory = tracker.track(sess, init_bb, filenames, video_log_dir)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)
#   
#       gt_bboxs = readbbox(osp.join(input_files, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---   IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
#       
#       T = 20
#       print ("dynamic branch t={}".format(T))
#       trajectory = tracker.track(sess, init_bb, filenames, video_log_dir, None, T)
#       #trajectory = tracker.track(sess, init_bb, filenames, video_log_dir)
#       with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
#         for region in trajectory:
#           rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
#                                             region.width, region.height)
#           f.write(rect_str)
#   
#       gt_bboxs = readbbox(osp.join(input_files, 'groundtruth_rect.txt'))
#       pred_bboxs = readbbox(osp.join(video_log_dir, 'track_rect.txt'))
#       print ("MulSiamFC class  ---   IOU --- {0}".format(cal_IOU(pred_bboxs, gt_bboxs)))
#       
# =============================================================================
      
      
      