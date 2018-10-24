#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: data.py

import cv2
import numpy as np
import copy
import glob
import time
from PIL import Image
from scipy.misc import imresize
import os

from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (
    MapData, MultiProcessMapData, PrefetchDataZMQ, imgaug, TestDataSpeed,
    MapDataComponent, DataFromList, RandomChooseData)
import tensorpack.utils.viz as tpviz

from coco import COCODetection, COCOMeta
from utils.generate_anchors import generate_anchors
#from utils.box_ops import get_iou_callable
from utils.np_box_ops import iou as np_iou
from common import (
    DataFromListOfDict, CustomResize,
    box_to_point8, point8_to_box, segmentation_to_mask)
import config


class MalformedData(BaseException):
    pass


@memoized
def get_all_anchors(
        stride=config.ANCHOR_STRIDE,
        sizes=config.ANCHOR_SIZES):
    """
    Get all anchors in the largest possible image, shifted, floatbox

    Returns:
        anchors: SxSxNUM_ANCHORx4, where S == MAX_SIZE//STRIDE, floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SCALE.

    """
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on stride / 2, have (approximate) sqrt areas of the specified
    # sizes, and aspect ratios as given.
    cell_anchors = generate_anchors(
        stride,
        scales=np.array(sizes, dtype=np.float) / stride,
        ratios=np.array(config.ANCHOR_RATIOS, dtype=np.float))
    # anchors are intbox here.
    # anchors at featuremap [0,0] are centered at fpcoor (8,8) (half of stride)

    field_size = config.MAX_SIZE // stride
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4)) +
        shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype('float32')
    field_of_anchors[:, :, :, [2, 3]] += 1
    return field_of_anchors


def get_anchor_labels(anchors, gt_boxes, crowd_boxes):
    """
    Label each anchor as fg/bg/ignore.
    Args:
        anchors: Ax4 float
        gt_boxes: Bx4 float
        crowd_boxes: Cx4 float

    Returns:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """
    # This function will modify labels and return the filtered inds
    def filter_box_label(labels, value, max_num):
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(
                curr_inds, size=(len(curr_inds) - max_num),
                replace=False)
            labels[disable_inds] = -1    # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    #bbox_iou_float = get_iou_callable()
    NA, NB = len(anchors), len(gt_boxes)
    assert NB > 0  # empty images should have been filtered already
    #box_ious = bbox_iou_float(anchors, gt_boxes)  # NA x NB
    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((NA,), dtype='int32')   # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= config.POSITIVE_ANCHOR_THRES] = 1
    anchor_labels[ious_max_per_anchor < config.NEGATIVE_ANCHOR_THRES] = 0

    # First label all non-ignore candidate boxes which overlap crowd as ignore
    if crowd_boxes.size > 0:
        cand_inds = np.where(anchor_labels >= 0)[0]
        cand_anchors = anchors[cand_inds]
        #ious = bbox_iou_float(cand_anchors, crowd_boxes)
        ious = np_iou(cand_anchors, crowd_boxes)
        overlap_with_crowd = cand_inds[ious.max(axis=1) > config.CROWD_OVERLAP_THRES]
        anchor_labels[overlap_with_crowd] = -1

    # Filter fg labels: ignore some fg if fg is too many
    target_num_fg = int(config.RPN_BATCH_PER_IM * config.RPN_FG_RATIO)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    # Note that fg could be fewer than the target ratio

    # filter bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0 or len(fg_inds) == 0:
        # No valid bg/fg in this image, skip.
        # This can happen if, e.g. the image has large crowd.
        raise MalformedData("No valid foreground/background for RPN!")
    target_num_bg = config.RPN_BATCH_PER_IM - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)   # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((NA, 4), dtype='float32')
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    return anchor_labels, anchor_boxes


def get_rpn_anchor_input(im, boxes, is_crowd):
    """
    Args:
        im: an image
        boxes: nx4, floatbox, gt. shoudn't be changed
        is_crowd: n,

    Returns:
        The anchor labels and target boxes for each pixel in the featuremap.
        fm_labels: fHxfWxNA
        fm_boxes: fHxfWxNAx4
    """
    boxes = boxes.copy()

    ALL_ANCHORS = get_all_anchors()
    H, W = im.shape[:2]
    featureH, featureW = H // config.ANCHOR_STRIDE, W // config.ANCHOR_STRIDE

    def filter_box_inside(im, boxes):
        h, w = im.shape[:2]
        indices = np.where(
            (boxes[:, 0] >= 0) &
            (boxes[:, 1] >= 0) &
            (boxes[:, 2] <= w) &
            (boxes[:, 3] <= h))[0]
        return indices

    crowd_boxes = boxes[is_crowd == 1]
    non_crowd_boxes = boxes[is_crowd == 0]

    # fHxfWxAx4
    featuremap_anchors = ALL_ANCHORS[:featureH, :featureW, :, :]
    featuremap_anchors_flatten = featuremap_anchors.reshape((-1, 4))
    # only use anchors inside the image
    inside_ind = filter_box_inside(im, featuremap_anchors_flatten)
    inside_anchors = featuremap_anchors_flatten[inside_ind, :]

    anchor_labels, anchor_boxes = get_anchor_labels(inside_anchors, non_crowd_boxes, crowd_boxes)

    # Fill them back to original size: fHxfWx1, fHxfWx4
    featuremap_labels = -np.ones((featureH * featureW * config.NUM_ANCHOR, ), dtype='int32')
    featuremap_labels[inside_ind] = anchor_labels
    featuremap_labels = featuremap_labels.reshape((featureH, featureW, config.NUM_ANCHOR))
    featuremap_boxes = np.zeros((featureH * featureW * config.NUM_ANCHOR, 4), dtype='float32')
    featuremap_boxes[inside_ind, :] = anchor_boxes
    featuremap_boxes = featuremap_boxes.reshape((featureH, featureW, config.NUM_ANCHOR, 4))
    return featuremap_labels, featuremap_boxes

def get_train_dataflow_davis(add_mask=False):
    # train_img_path = config.DAVIS_PATH + "train/"
    # train_label_path = config.DAVIS_PATH + "train-gt/"
    # imgs = glob.glob(train_img_path + "*/*.jpg")

    # train_img_path = "/home/luiten/vision/PReMVOS/data/first/bike-trial/lucid_data_dreaming/"
    # train_label_path = "/home/luiten/vision/PReMVOS/data/first/bike-trial/lucid_data_dreaming/"

    # train_img_path = "/home/luiten/vision/PReMVOS/data/"+config.DAVIS_NAME+"/lucid_data_dreaming/"
    # train_label_path = "/home/luiten/vision/PReMVOS/data/"+config.DAVIS_NAME+"/lucid_data_dreaming/"

    # train_img_path = "/home/luiten/vision/youtubevos/ytvos_data/together/generated/augment_images/"
    # train_label_path = "/home/luiten/vision/youtubevos/ytvos_data/together/generated/augment_gt/"

    train_img_path = "/home/luiten/vision/youtubevos/DAVIS/davis_together/augment_images/"
    train_label_path = "/home/luiten/vision/youtubevos/DAVIS/davis_together/augment_gt/"


    imgs = sorted(glob.glob(train_img_path + "*/*.jpg"))

    ds = DataFromList(imgs, shuffle=True)
    aug = imgaug.AugmentorList(
        [CustomResize(config.SHORT_EDGE_SIZE, config.MAX_SIZE),
         imgaug.Flip(horiz=True)])

    def preprocess(fname):
        # print("start preproc mapillary")
        start = time.time()

        label_fname = fname.replace(train_img_path, train_label_path).replace(".jpg", ".png")
        pil_label = Image.open(label_fname)
        label = np.array(pil_label)
        instances = np.unique(label)
        instance_classes = [x // 256 for x in instances]

        if len(instances) == 0:
            print("no instances")
            pil_label.close()
            return None

        masks = np.array([label == inst for inst in instances], dtype=np.uint8)

        boxes1 = np.array([get_bbox_from_segmentation_mask(mask) for mask in masks], dtype=np.float32)
        boxes = boxes1

        # second_klass = np.array(instance_classes, dtype=np.int)
        second_klass = np.zeros_like(instance_classes, dtype=np.int)
        klass = np.ones_like(second_klass)
        is_crowd = np.zeros_like(second_klass)

        res = preproc_img(fname, boxes, klass, second_klass, is_crowd, aug)
        if res is None:
            print("davis: preproc_img returned None on", fname)
            pil_label.close()
            return None
        ret, params = res
        if add_mask:
            do_flip, h, w = params[1]
            assert do_flip in (True, False), do_flip
            # augment label
            label = np.array(pil_label.resize((w, h), Image.NEAREST))
            if do_flip:
                label = label[:, ::-1]
            # create augmented masks
            masks = np.array([label == inst for inst in instances], dtype=np.uint8)
            ret.append(masks)

        end = time.time()
        elapsed = end - start
        # print("davis example done, elapsed:", elapsed)

        VISUALIZE = False
        if VISUALIZE:
            from viz import draw_annotation, draw_mask
            config.CLASS_NAMES = [str(idx) for idx in range(81)]
            im = ret[0]
            boxes = ret[3]
            draw_klass = ret[-2]
            viz = draw_annotation(im, boxes, draw_klass)
            for mask in masks:
                viz = draw_mask(viz, mask)
            tpviz.interactive_imshow(viz)

        pil_label.close()
        return ret

    ds = MapData(ds, preprocess)
    # ds = MultiProcessMapData(ds, nr_proc=8, map_func=preprocess, buffer_size=35)
    # ds = MultiProcessMapData(ds, nr_proc=8, map_func=preprocess)
    return ds

def get_train_dataflow_mapillary(add_mask=False, map_to_coco=False):
    train_img_path = config.MAPILLARY_PATH + "training/images/"
    train_label_path = config.MAPILLARY_PATH + "training/instances/"
    imgs = glob.glob(train_img_path + "*.jpg")

    ds = DataFromList(imgs, shuffle=True)
    aug = imgaug.AugmentorList(
        [CustomResize(config.SHORT_EDGE_SIZE, config.MAX_SIZE),
         imgaug.Flip(horiz=True)])

    def preprocess(fname):
        print("start preproc mapillary")
        start = time.time()

        label_fname = fname.replace(train_img_path, train_label_path).replace(".jpg", ".png")
        pil_label = Image.open(label_fname)
        label = np.array(pil_label)
        instances = np.unique(label)
        instance_classes = [x // 256 for x in instances]

        # filter by categories we use
        instances_valid = [cls in config.MAPILLARY_CAT_IDS_TO_USE for cls in instance_classes]
        instances = [inst for inst, valid in zip(instances, instances_valid) if valid]
        instance_classes = [cls for cls, valid in zip(instance_classes, instances_valid) if valid]

        if len(instances) == 0:
            print("no instances")
            pil_label.close()
            return None

        if map_to_coco:
            instance_classes = [config.MAPILLARY_TO_COCO_MAP[cls] for cls in instance_classes]
            instance_classes = [config.VOID_LABEL if cls == config.VOID_LABEL else COCOMeta.category_id_to_class_id[cls]
                                for cls in instance_classes]
        else:
            # remap to contiguous numbers starting with 1
            instance_classes = [config.MAPILLARY_CAT_IDS_TO_USE.index(cls) + 1 for cls in instance_classes]

        masks = np.array([label == inst for inst in instances], dtype=np.uint8)

        #import cProfile
        #start1 = time.time()
        boxes1 = np.array([get_bbox_from_segmentation_mask(mask) for mask in masks], dtype=np.float32)
        #boxes1_time = time.time() - start1
        #pr = cProfile.Profile()
        #pr.enable()
        #start1 = time.time()
        #boxes2 = get_bboxes_from_segmentation_masks(masks)
        #print("boxes1", boxes1_time, "boxes2", time.time() - start1)
        #pr.disable()
        #pr.print_stats(sort="cumulative")
        #assert (boxes1 == boxes2).all(), (boxes1, boxes2)
        boxes = boxes1

        second_klass = np.array(instance_classes, dtype=np.int)
        klass = np.ones_like(second_klass)
        is_crowd = np.zeros_like(second_klass)

        res = preproc_img(fname, boxes, klass, second_klass, is_crowd, aug)
        if res is None:
            print("mapillary: preproc_img returned None on", fname)
            pil_label.close()
            return None
        ret, params = res
        if add_mask:
            do_flip, h, w = params[1]
            assert do_flip in (True, False), do_flip
            # augment label
            label = np.array(pil_label.resize((w, h), Image.NEAREST))
            if do_flip:
                label = label[:, ::-1]
            # create augmented masks
            masks = np.array([label == inst for inst in instances], dtype=np.uint8)
            ret.append(masks)

        end = time.time()
        elapsed = end - start
        print("mapillary example done, elapsed:", elapsed)

        VISUALIZE = False
        if VISUALIZE:
            from viz import draw_annotation, draw_mask
            config.CLASS_NAMES = [str(idx) for idx in range(81)]
            im = ret[0]
            boxes = ret[3]
            draw_klass = ret[-2]
            viz = draw_annotation(im, boxes, draw_klass)
            for mask in masks:
                viz = draw_mask(viz, mask)
            tpviz.interactive_imshow(viz)

        pil_label.close()
        return ret

    #ds = MapData(ds, preprocess)
    ds = MultiProcessMapData(ds, nr_proc=8, map_func=preprocess, buffer_size=35)
    return ds


def get_train_dataflow_coco_and_mapillary(add_mask=False):
    dataflow_coco = get_train_dataflow_coco(add_mask)
    dataflow_mapillary = get_train_dataflow_mapillary(add_mask, map_to_coco=True)
    dataflow_combined = RandomChooseData([dataflow_coco, dataflow_mapillary])
    return dataflow_combined


def get_bboxes_from_segmentation_masks(masks):
    rows = np.any(masks, axis=2)
    cols = np.any(masks, axis=1)
    y0 = np.argmax(rows, axis=1)
    x0 = np.argmax(cols, axis=1)
    y1 = rows.shape[1] - np.argmax(rows[:, ::-1], axis=1)
    x1 = cols.shape[1] - np.argmax(cols[:, ::-1], axis=1)
    return np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)


def get_bbox_from_segmentation_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y1 += 1
    x1 += 1
    bbox = np.array([x0, y0, x1, y1], dtype=np.float32)
    return bbox


def get_train_dataflow_coco(add_mask=False):
    """
    Return a training dataflow. Each datapoint is:
    image, fm_labels, fm_boxes, gt_boxes, gt_class [, masks]
    """
    imgs = COCODetection.load_many(
        config.BASEDIR, config.TRAIN_DATASET, add_gt=True, add_mask=add_mask)
    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    imgs = list(filter(lambda img: len(img['boxes']) > 0, imgs))    # log invalid training

    ds = DataFromList(imgs, shuffle=True)

    aug = imgaug.AugmentorList(
        [CustomResize(config.SHORT_EDGE_SIZE, config.MAX_SIZE),
         imgaug.Flip(horiz=True)])

    def preprocess(img):
        print("start preproc coco")
        start = time.time()
        if config.USE_SECOND_HEAD:
            fname, boxes, klass, second_klass, is_crowd = img['file_name'], img['boxes'], img['class'], \
                                                          img['second_class'], img['is_crowd']
        else:
            fname, boxes, klass, is_crowd = img['file_name'], img['boxes'], img['class'], img['is_crowd']
            second_klass = None
        res = preproc_img(fname, boxes, klass, second_klass, is_crowd, aug)
        if res is None:
            print("coco: preproc_img returned None on", fname)
            return None

        ret, params = res
        im = ret[0]
        boxes = ret[3]
        # masks
        if add_mask:
            # augmentation will modify the polys in-place
            segmentation = copy.deepcopy(img.get('segmentation', None))
            segmentation = [segmentation[k] for k in range(len(segmentation)) if not is_crowd[k]]
            assert len(segmentation) == len(boxes), (len(segmentation), len(boxes))

            # one image-sized binary mask per box
            masks = []
            for polys in segmentation:
                polys = [aug.augment_coords(p, params) for p in polys]
                masks.append(segmentation_to_mask(polys, im.shape[0], im.shape[1]))
            masks = np.asarray(masks, dtype='uint8')    # values in {0, 1}
            ret.append(masks)

            # from viz import draw_annotation, draw_mask
            # viz = draw_annotation(im, boxes, klass)
            # for mask in masks:
            #     viz = draw_mask(viz, mask)
            # tpviz.interactive_imshow(viz)
        end = time.time()
        elapsed = end - start
        print("coco example done, elapsed:", elapsed)
        return ret

    #ds = MapData(ds, preprocess)
    ds = MultiProcessMapData(ds, nr_proc=4, map_func=preprocess, buffer_size=20)
    return ds


def preproc_img(fname, boxes, klass, second_klass, is_crowd, aug):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    assert im is not None, fname
    im = im.astype('float32')
    # assume floatbox as input
    assert boxes.dtype == np.float32

    # augmentation:
    im, params = aug.augment_return_params(im)
    points = box_to_point8(boxes)
    points = aug.augment_coords(points, params)
    boxes = point8_to_box(points)

    # rpn anchor:
    try:
        fm_labels, fm_boxes = get_rpn_anchor_input(im, boxes, is_crowd)
        boxes = boxes[is_crowd == 0]  # skip crowd boxes in training target
        klass = klass[is_crowd == 0]
        if config.USE_SECOND_HEAD:
            second_klass = second_klass[is_crowd == 0]
        if not len(boxes):
            raise MalformedData("No valid gt_boxes!")
    except MalformedData as e:
        log_once("Input {} is filtered for training: {}".format(fname, str(e)), 'warn')
        return None

    if config.USE_SECOND_HEAD:
        ret = [im, fm_labels, fm_boxes, boxes, klass, second_klass]
    else:
        ret = [im, fm_labels, fm_boxes, boxes, klass]
    return ret, params


def get_eval_dataflow():
    imgs = COCODetection.load_many(config.BASEDIR, config.VAL_DATASET, add_gt=False)
    # no filter for training
    ds = DataFromListOfDict(imgs, ['file_name', 'id'])

    def f(fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        return im
    ds = MapDataComponent(ds, f, 0)
    # ds = PrefetchDataZMQ(ds, 1)
    return ds


if __name__ == '__main__':
    config.BASEDIR = '/home/wyx/data/coco'
    config.TRAIN_DATASET = ['train2014']
    from tensorpack.dataflow import PrintData
    ds = get_train_dataflow_coco()
    ds = PrintData(ds, 100)
    TestDataSpeed(ds, 50000).start()
    ds.reset_state()
    for k in ds.get_data():
        pass
