#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py
# Search for #YTVOS-EDIT

import numpy as np
import os

# added by Paul
CATEGORY_AGNOSTIC = False
TRAIN_HEADS_ONLY = False

#added by Jono / Paul
USE_SECOND_HEAD = False
USE_MAPILLARY = False
USE_COCO_AND_MAPILLARY = False
USE_DAVIS = False

MAPILLARY_PATH = '/fastwork/' + os.environ["USER"] + "/mywork/data/mapillary_quarter/"
DAVIS_NAME = ""

vehicle_ids = [52, 53, 54, 55, 56, 57, 59, 60, 61, 62]
human_ids = [19, 20, 21, 22]
animal_ids = [0, 1]
object_ids = [32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 48, 49, 50, 51]
crosswalk_zebra_id = [23]
MAPILLARY_CAT_IDS_TO_USE = vehicle_ids + human_ids + animal_ids + object_ids + crosswalk_zebra_id
VOID_LABEL = 255
MAPILLARY_TO_COCO_MAP = {0: 16, 1: VOID_LABEL, 8: VOID_LABEL, 19: 1, 20: 1, 21: 1, 22: 1, 23: VOID_LABEL,
                         32: VOID_LABEL, 33: 15, 34: VOID_LABEL, 35: VOID_LABEL, 36: VOID_LABEL,
                         37: VOID_LABEL, 38: 11, 39: VOID_LABEL, 40: VOID_LABEL, 41: VOID_LABEL,
                         42: VOID_LABEL, 44: VOID_LABEL, 45: VOID_LABEL, 46: VOID_LABEL,
                         47: VOID_LABEL, 48: 10, 49: VOID_LABEL, 50: VOID_LABEL, 51: VOID_LABEL,
                         52: 2, 53: 9, 54: 6, 55: 3, 56: VOID_LABEL, 57: 4, 59: VOID_LABEL,
                         60: VOID_LABEL, 61: 8, 62: VOID_LABEL}

EXTRACT_FEATURES = False
PROVIDE_BOXES_AS_INPUT = False

# mode flags ---------------------
#YTVOS-EDIT (uneddited back to True for training, will need to change again for forwarding)
# MODE_MASK = True
#MODE_MASK = False

### IN ORDER TO NOT MAKE THIS MISTAKE AGAIN: MODE_MASK_NOW NOT DEFINED HERE BUT DEFINED IN TRAIN.PY DEPENDING ON TRAIN/FORWARD

# dataset -----------------------
BASEDIR = '/fastwork/' + os.environ["USER"] + '/mywork/data/coco/'
TRAIN_DATASET = ['train2014', 'valminusminival2014']
VAL_DATASET = 'minival2014'   # only support evaluation on single dataset
if CATEGORY_AGNOSTIC:
    NUM_CLASS = 2
else:
    NUM_CLASS = 81
SECOND_NUM_CLASS = 81
CLASS_NAMES = []  # NUM_CLASS strings
SECOND_CLASS_NAMES = []  # NUM_CLASS strings

# basemodel ----------------------
#RESNET_NUM_BLOCK = [3, 4, 6, 3]     # resnet50
RESNET_NUM_BLOCK = [3, 4, 23, 3]     # resnet101

# preprocessing --------------------
SHORT_EDGE_SIZE = 800
MAX_SIZE = 1333
# alternative (worse & faster) setting: 600, 1024

# anchors -------------------------
ANCHOR_STRIDE = 16
# sqrtarea of the anchor box
ANCHOR_SIZES = (32, 64, 128, 256, 512)
ANCHOR_RATIOS = (0.5, 1., 2.)
NUM_ANCHOR = len(ANCHOR_SIZES) * len(ANCHOR_RATIOS)
POSITIVE_ANCHOR_THRES = 0.7
NEGATIVE_ANCHOR_THRES = 0.3
# just to avoid too large numbers.
BBOX_DECODE_CLIP = np.log(MAX_SIZE / 16.0)

# rpn training -------------------------
# fg ratio among selected RPN anchors
RPN_FG_RATIO = 0.5
RPN_BATCH_PER_IM = 256
RPN_MIN_SIZE = 0
RPN_PROPOSAL_NMS_THRESH = 0.7
TRAIN_PRE_NMS_TOPK = 12000
TRAIN_POST_NMS_TOPK = 2000
# boxes overlapping crowd will be ignored.
CROWD_OVERLAP_THRES = 0.7

# fastrcnn training ---------------------
FASTRCNN_BATCH_PER_IM = 256
FASTRCNN_BBOX_REG_WEIGHTS = np.array([10, 10, 5, 5], dtype='float32')
FASTRCNN_FG_THRESH = 0.5
# fg ratio in a ROI batch
FASTRCNN_FG_RATIO = 0.25

# testing -----------------------
#TEST_PRE_NMS_TOPK = 6000
#YTVOS-EDIT
#TEST_PRE_NMS_TOPK = 15000
TEST_PRE_NMS_TOPK = 1000

#TEST_POST_NMS_TOPK = 1000   # if you encounter OOM in inference, set this to a smaller number
#YTVOS-EDIT
# TEST_POST_NMS_TOPK = 1000
TEST_POST_NMS_TOPK = 100

#FASTRCNN_NMS_THRESH = 0.5
#YTVOS-EDIT
# FASTRCNN_NMS_THRESH = 0.8
FASTRCNN_NMS_THRESH = 0.5

#RESULT_SCORE_THRESH = 0.05
#YTVOS-EDIT
# RESULT_SCORE_THRESH = 0.0
#RESULT_SCORE_THRESH = 0.1
RESULT_SCORE_THRESH = 0.5

#RESULTS_PER_IM = 100
#YTVOS-EDIT
# RESULTS_PER_IM = 1000
#RESULTS_PER_IM = 50
RESULTS_PER_IM = 20
