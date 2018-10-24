#!/usr/bin/env python3
from multiprocessing.pool import Pool
from functools import partial

import numpy as np
import os
from pycocotools.mask import encode, iou, area, decode, merge,toBbox
import json
import glob
from PIL import Image
from numpy.linalg import norm
from numpy import array as arr
from copy import deepcopy as copy
import time

from MergeTrack.merge_functions import save_with_pascal_colormap, eval_video

# to_do = "DAVIS"
# to_do = "ytvos_data/valid"
# to_do = "DAVIS_new/val-test"
to_do = "DAVIS_new/val-old"

# image_dir = "/home/luiten/vision/youtubevos/%s/hard_subset/val17/"%to_do
# # image_dir = "/home/luiten/vision/youtubevos/%s/JPEGImages/"%to_do
# proposal_dir = "/home/luiten/vision/youtubevos/%s/flow_props-final/"%to_do
# ff_dir = "/home/luiten/vision/youtubevos/%s/ff_props/"%to_do
# gt_dir = "/home/luiten/vision/youtubevos/%s/val17-gt/"%to_do
# ensemble_output_dir = "/home/luiten/vision/youtubevos/%s/to_ensemble_old_merge/"%to_do
# viz_dir = "/home/luiten/vision/youtubevos/%s/viz2/"%to_do
# hyperopt_file_name = "/home/luiten/vision/youtubevos/%s/hyperopt-old_merge/"%to_do +str(np.random.randint(0,2**52))+'.txt'

image_dir = "/home/luiten/vision/youtubevos/%s/images/"%to_do
proposal_dir = "/home/luiten/vision/youtubevos/%s/final-props/"%to_do
ff_dir = "/home/luiten/vision/youtubevos/%s/ff_props/"%to_do
gt_dir = "/home/luiten/vision/youtubevos/%s/gt/"%to_do

ensemble_output_dir = "/home/luiten/vision/youtubevos/%s/to_ensemble/"%to_do
debug_viz_dir = "/home/luiten/vision/youtubevos/%s/debug_viz/"%to_do
hyperopt_file_name = "/home/luiten/vision/youtubevos/%s/hyperopt/"%to_do +str(np.random.randint(0,2**52))+'.txt'

optimize = False
eval = True
use_ff_negative_props = False

curr_run_num = 0
total_to_run = 1

def read_all_props(image_fn_list,curr_dir):
  files = [f.replace(image_dir,curr_dir).replace('.jpg','.json') for f in image_fn_list]
  all_props = []
  for file in files:
    try:
      with open(file, "r") as f:
        proposals = json.load(f)
      for prop in proposals:
        if 'ReID' not in prop.keys():
          prop['ReID'] = np.inf * np.ones((128))
    except:
      proposals = []
    all_props.append(proposals)
  return all_props

def get_negative_ff_props(proposals,templates):
  if use_ff_negative_props:
    curr_propsoals = copy(proposals)
    curr_set = copy(templates)
    negatives = []

    curr_propsoals.sort(key=lambda x: x["score"], reverse=True)

    for prop in curr_propsoals:
      is_ol = False
      for s in curr_set:
        ol = iou([s['segmentation'],],[prop['segmentation'],],np.array([0],np.uint8))
        if ol>0:
          is_ol = True
          break
      if is_ol:
        continue
      curr_set+= copy([prop,])
      negatives+=copy([prop,])
  else:
    negatives = []

  return negatives

def get_all_reid_scores(all_props,templates):

  all_reid_distances = [arr([[norm(np.array(prop['ReID']) - np.array(templ['ReID'])) for prop in props] for templ in templates]) for props in all_props]
  all_reid_distances_no_inf = copy(all_reid_distances)

  for mat in all_reid_distances_no_inf:
    mat[np.isinf(mat)] = 0
  max_distances = arr([mat.max(axis=1) if mat.shape[1]>0 else np.zeros((mat.shape[0])) for mat in all_reid_distances_no_inf]).max(axis=0)
  max_distances = [np.repeat(max_distances[:, np.newaxis], mat.shape[1], axis=1) for mat in all_reid_distances_no_inf]
  all_reid_scores = [1 - mat/max_dist for mat,max_dist in zip(all_reid_distances,max_distances)]
  for mat in all_reid_scores:
    mat[np.isinf(mat)] = 0

  all_other_reid_scores = []
  for mat in all_reid_scores:
    new_mat = np.ones_like(mat)
    if len(templates)>1:
      ids = np.arange(len(templates))
      for id in ids:
        new_vec = 1 - np.max(np.atleast_2d(mat[ids!=id,:]),axis=0)
        new_mat[id,:] = new_vec
    all_other_reid_scores.append(new_mat)

  return all_reid_scores,all_other_reid_scores\

def calculate_old_merge_scores(proposals, templates,next_props,reid_scores,other_reid_scores):
  mask_scores = np.repeat(np.array([prop['score'] for prop in proposals])[np.newaxis, :],len(templates),axis=0)
  segs = [prop["segmentation"] for prop in proposals]
  next_segs = [prop["segmentation"] for prop in next_props]
  warp_scores = arr([iou(segs, [next_seg, ], arr([0], np.uint8))[:, 0] for next_seg in next_segs])

  other_warp_scores = np.ones_like(warp_scores)
  if len(templates) > 1:
    ids = np.arange(len(templates))
    for id in ids:
      new_vec = 1 - np.max(np.atleast_2d(warp_scores[ids != id, :]), axis=0)
      other_warp_scores[id, :] = new_vec

  all_scores = arr([mask_scores, reid_scores, other_reid_scores,
                    warp_scores, other_warp_scores])
  return all_scores

def do_video(video_fn,ensemble_num):
  output_image_dir = ensemble_output_dir + str(ensemble_num) + '/'
  print("Starting", video_fn)
  final_solution = []

  image_fn_list = sorted(glob.glob(video_fn + "*.jpg"))
  all_props = read_all_props(image_fn_list,proposal_dir)
  all_ff_props = read_all_props(image_fn_list,ff_dir)
  negative_ff_props = get_negative_ff_props(all_props[0], all_ff_props[0])
  templates = [prop for props in all_ff_props for prop in props] + negative_ff_props
  all_reid_scores, all_other_reid_scores = get_all_reid_scores(all_props,templates)

  empty_prop = dict()
  size = templates[0]['segmentation']['size']
  empty_seg = encode(np.asfortranarray(np.zeros(size, dtype=np.uint8)))
  empty_seg['counts'] = empty_seg['counts'].decode("utf-8")
  empty_prop['segmentation'] = empty_seg

  next_props = all_ff_props[0] + [copy(empty_prop) for props in all_ff_props[1:] for _ in props] + negative_ff_props
  labels = [0 for _ in templates]

  for image_id, image_fn in enumerate(image_fn_list):
    proposals = all_props[image_id]
    all_scores = calculate_old_merge_scores(proposals, templates,next_props,all_reid_scores[image_id],all_other_reid_scores[image_id])
    weighted_scores = np.dot(normalised_weights, all_scores.transpose((1, 0, 2)), )

    do_snapping = True
    if do_snapping:
      closest_gts = np.argmax(weighted_scores, axis=0)
      snapped_scores = weighted_scores
      for gt_id in range(len(templates)):
        snapped_scores[gt_id,:] = snapped_scores[gt_id,:]*(closest_gts==gt_id)
    else:
      snapped_scores = weighted_scores

    best_scores = snapped_scores.max(axis=1)
    best_scores_index = snapped_scores.argmax(axis=1)
    chosen_props = [proposals[best_scores_index[i]] for i in range(len(templates))]

    if all_ff_props[image_id]:
      before_ff_props = [prop for props in all_ff_props[:image_id] for prop in props]
      ii = np.arange(len(before_ff_props),len(before_ff_props)+len(all_ff_props[image_id]))
      for i in ii:
        chosen_props[i] = copy(templates[i])
        labels[i] = chosen_props[i]['id']
        best_scores[i] = 1.0

    proposal_order = np.argsort(best_scores)[::-1]
    chosen_masks = [decode(prop['segmentation']) for prop in chosen_props]

    png_index = np.zeros_like(chosen_masks[0])
    for i in proposal_order[::-1]:
      png_index[chosen_masks[i].astype("bool")] = i+1

    refined_masks = [(png_index == i+1).astype(np.uint8) for i in range(len(templates))]

    final_props = []
    refined_segmentations = [encode(np.asfortranarray(mask)) if mask is not None else copy(empty_seg) for mask in refined_masks]
    for id_,seg in enumerate(refined_segmentations):
      seg['counts'] = seg['counts'].decode("utf-8")
      prop = dict()
      prop['segmentation'] = seg
      final_props.append(prop)

      # Need to warp, can warp live, no need for pre-warp
      # Or I guess two options here, live warp or pre-warp. PREMVOS 1 uses pre-warp, we use that now for compatibility,
      # but think live warp will work better
      if image_id != len(image_fn_list)-1:
        next_props[id_]['segmentation'] = chosen_props[id_]['forward_segmentation']

    if not optimize:
      png_labels = np.zeros_like(chosen_masks[0])
      for i in range(len(templates)):
        png_labels[png_index==i+1] = labels[i]

      output_fn = image_fn.replace(image_dir,output_image_dir).replace('.jpg','.png')
      output_dir = '/'.join(output_fn.split('/')[:-1])+'/'
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      save_with_pascal_colormap(output_fn, png_labels)

    final_solution.append((image_fn, final_props[:len(all_ff_props[0])]))

  if to_do == "DAVIS":
    vid_scores = eval_video(final_solution, image_dir, gt_dir)
    print(video_fn.split('/')[-1], vid_scores)
  else:
    vid_scores = None

  return vid_scores

weight_set = [
      [0.1639026729185494, 0.3090363324359478, 0.11728252456485666, 0.18345061062541546, 0.2263278594552307]]

for ensemble_num,c_weights in enumerate(weight_set):
  weights = np.array(c_weights)
# while(True):
#   weights = np.random.random_sample(5)
#   ensemble_num = 0

  normalised_weights = weights / np.sum(weights)

  final_scores = []
  video_list = sorted(glob.glob(image_dir+"*/"))
  num_per = int(np.ceil(len(video_list)/total_to_run))
  print(curr_run_num,num_per,curr_run_num*num_per,(curr_run_num+1)*num_per)
  video_list = video_list[curr_run_num*num_per : (curr_run_num+1)*num_per]

  # _do_video = partial(do_video,ensemble_num=ensemble_num)
  # with Pool(8) as pool:
  #   final_scores = pool.map(_do_video, video_list)

  start_time = time.time()
  for video_fn in video_list:
    vid_scores = do_video(video_fn,ensemble_num)
    final_scores.append(vid_scores)
  print(time.time()-start_time)

  if to_do == "DAVIS":
    for to_print in final_scores:
      print(to_print)
    list_scores = [score for scores in final_scores for score in scores]
    print("final score:", np.mean(list_scores))

  if optimize:
    with open(hyperopt_file_name, "a") as f:
      print(np.mean(list_scores)*100, ' '.join(map(str, normalised_weights)), ' '.join(map(str, list_scores)), file=f, sep=" ", end="\n")