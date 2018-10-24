import json
import numpy as np
from PIL import Image
from pycocotools.mask import encode, iou, area, decode, toBbox, merge
from numpy.linalg import norm
import cv2
import os
from tensorpack.utils.palette import PALETTE_RGB
from scipy.misc import imread,imsave
from copy import deepcopy as copy

MAX_REID_DISTANCE = 25 # Can be approximated from print_max_reid_distance.py

def read_ann(ann_fn):
  ann = np.array(Image.open(ann_fn))
  ids = np.unique(ann)
  ids = [id for id in ids if id != 0]
  ann_masks = [(ann == id_).astype(np.uint8) for id_ in ids if id_ != 0]
  new_proposals = []
  for ann_mask,id in zip(ann_masks,ids):
    encoded_mask = encode(np.asfortranarray(ann_mask))
    encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")
    bbox = toBbox(encoded_mask)
    new_proposals.append({'id':id, 'bbox': bbox, 'segmentation': encoded_mask,'conf_score':"1.0",'score':1.0})
  return new_proposals

def read_props(prop_fn):
  try:
    with open(prop_fn, "r") as f:
      proposals = json.load(f)
    for prop in proposals:
      if 'ReID' not in prop.keys():
        prop['ReID'] = np.inf * np.ones((128))
  except:
    proposals = []
  return proposals

def calculate_scores(proposals,templates):
  segs = [prop['segmentation'] for prop in proposals]
  template_segs = [templ['segmentation'] for templ in templates]
  warp_scores = np.array([iou(segs, [template_seg, ], np.array([0], np.uint8))[:, 0] for template_seg in template_segs])
  warped_score_weights = np.array([templ['score'] for templ in templates])[:,np.newaxis]
  warped_score_min = 0.5
  warped_score_weights = np.maximum(warped_score_weights - warped_score_min, 0) / (1 - warped_score_min)
  warp_scores = warp_scores*warped_score_weights

  ReID_embs = [prop['ReID'] for prop in proposals]
  template_ReID_embs = [templ['ReID'] for templ in templates]

  ReID_distances = np.array([[norm(np.array(c_reid) - np.array(gt_reid)) for c_reid in ReID_embs] for gt_reid in template_ReID_embs])
  ReID_scores = 1 - ReID_distances/MAX_REID_DISTANCE
  ReID_scores[np.isinf(ReID_scores)] = 0
  ReID_scores[np.less(ReID_scores,0)] = 0

  other_warp_scores = np.ones_like(warp_scores)
  other_ReID_scores = np.ones_like(ReID_scores)
  if len(templates) > 1:
    ids = np.indices([len(templates)])[0]
    for id in ids:
      new_warp_vec = 1 - np.max(np.atleast_2d(warp_scores[ids != id, :]), axis=0)
      other_warp_scores[id, :] = new_warp_vec
      new_ReID_vec = 1 - np.max(np.atleast_2d(ReID_scores[ids != id, :]), axis=0)
      other_ReID_scores[id, :] = new_ReID_vec

  # mask_scores = np.ones_like(warp_scores)

  # mask_scores = np.repeat(np.array([float(prop['conf_score']) for prop in proposals])[np.newaxis, :],warp_scores.shape[0],axis=0)
  # mask_score_min = 0.8
  # mask_scores = np.maximum(mask_scores-mask_score_min,0) / (1-mask_score_min)

  mask_scores = np.repeat(np.array([float(prop['score']) for prop in proposals])[np.newaxis, :],warp_scores.shape[0], axis=0)
  mask_score_min = 0.5
  mask_scores = np.maximum(mask_scores - mask_score_min, 0) / (1 - mask_score_min)

  scores = np.array([mask_scores, ReID_scores, other_ReID_scores,warp_scores, other_warp_scores])
  return scores

def calculate_gt_scores(proposals,gt_props,templates):
  segs = [prop['segmentation'] for prop in proposals]

  empty_mask = np.zeros_like(decode(segs[0]))
  empty_seg = encode(np.asfortranarray(empty_mask))
  empty_seg['counts'] = empty_seg['counts'].decode("utf-8")

  gt_segs_init = [templ['segmentation'] for templ in gt_props]
  gt_ids_init = [templ['id'] for templ in gt_props]

  gt_segs = [empty_seg for _ in templates]
  for init_seg,init_id in zip(gt_segs_init,gt_ids_init):
    gt_segs[init_id-1] = init_seg

  gt_scores = np.array([iou(segs, [gt_seg, ], np.array([0], np.uint8))[:, 0] for gt_seg in gt_segs])
  scores = np.array([gt_scores, gt_scores, gt_scores, gt_scores, gt_scores])
  return scores

def calculate_selected_props(proposals,weighted_scores,templates,score_thresh,object_scores):

  empty_mask = np.zeros_like(decode(proposals[0]['segmentation']))
  empty_seg = encode(np.asfortranarray(empty_mask))
  empty_seg['counts'] = empty_seg['counts'].decode("utf-8")
  empty_prop = dict()
  empty_prop['segmentation'] = empty_seg
  empty_prop['bbox'] = toBbox(empty_seg)

  proposals.append(empty_prop)
  # print(weighted_scores.shape,(score_thresh*np.ones_like(weighted_scores[:,0])).shape)
  weighted_scores = np.append(weighted_scores,score_thresh*np.ones((weighted_scores.shape[0],1)), axis=1)

  weighted_scores[np.logical_not(np.isfinite(weighted_scores))] = 0

  best_scores = weighted_scores.max(axis=1)
  best_scores_index = weighted_scores.argmax(axis=1)
  best_object_scores = object_scores.max(axis=1)

  selected_props = [proposals[i].copy() for i in best_scores_index]
  for order_id,(prop,score,template,object_score) in enumerate(zip(selected_props,best_scores,templates,best_object_scores)):
    prop['final_score'] = score
    prop['object_score'] = object_score
    prop['id'] = template['id'] #todo Check if the ID here works
    # prop['id'] = order_id
  return selected_props

def remove_mask_overlap(proposals):
  scores = [prop['final_score'] if prop['final_score'] else 0 for prop in proposals]
  masks = [decode(prop['segmentation']) for prop in proposals]
  object_scores = [prop['object_score'] if prop['object_score'] else 0 for prop in proposals]
  ids = [prop['id'] for prop in proposals]
  proposal_order = np.argsort(scores)[::-1]
  # proposal_order = np.argsort(scores)
  labels = np.arange(1, len(proposal_order) + 1)
  png = np.zeros_like(masks[0])
  for i in proposal_order[::-1]:
    png[masks[i].astype("bool")] = labels[i]

  refined_masks = [(png == id_).astype(np.uint8) for id_ in labels]
  refined_segmentations = [encode(np.asfortranarray(refined_mask)) for refined_mask in refined_masks]
  selected_props = []
  for refined_segmentation,score,mask,id,object_score in zip(refined_segmentations,scores,refined_masks,ids,object_scores):
    refined_segmentation['counts'] = refined_segmentation['counts'].decode("utf-8")
    prop = dict()
    prop['segmentation']=refined_segmentation
    prop['bbox']=toBbox(refined_segmentation)
    prop['final_score'] = score
    prop['object_score'] = object_score
    prop['mask'] = mask
    prop['id'] = id
    selected_props.append(prop)

  return selected_props

def probabilitic_combination(proposals,weighted_scores,templates,score_thesh):

  temperature = 0.1

  # Turn scores into probabilities over proposals using softmax
  probs_over_proposals = np.exp(weighted_scores/temperature)
  partition_function = probs_over_proposals.sum(axis=1)[:,np.newaxis]
  probs_over_proposals = probs_over_proposals/partition_function

  print(probs_over_proposals)

  masks = np.array([decode(prop['segmentation']) for prop in proposals]).astype(np.float32)
  masks = np.repeat(masks[np.newaxis,:,:,:],[probs_over_proposals.shape[0]], axis=0)
  # combined_masks = masks.mean(axis=1)
  probs_over_proposals_expanded = np.repeat(np.repeat(probs_over_proposals[:,:,np.newaxis], [masks.shape[2]],axis=2)[:,:,:,np.newaxis],masks.shape[3],axis=3)
  combined_masks = np.average(masks,axis=1,weights=probs_over_proposals_expanded)
  prob_any_obj = combined_masks.max(axis=0)[np.newaxis,:,:]
  prob_background = np.ones_like(prob_any_obj) - prob_any_obj
  probs_incl_background = np.append(prob_background,combined_masks,axis=0)
  probs_over_objects = np.exp(probs_incl_background/temperature)

  # print(probs_over_proposals.shape,partition_function.shape,masks.shape,probs_over_proposals_expanded.shape,
  #       combined_masks.shape,prob_any_obj.shape,prob_background.shape,probs_incl_background.shape,probs_over_objects.shape)

  partition_function_2 = probs_over_objects.sum(axis=0)[np.newaxis,:,:]
  probs_over_objects = probs_over_objects/partition_function_2
  final_results = np.argmax(probs_over_objects,axis=0)
  labels = np.arange(1, weighted_scores.shape[0]+1)
  final_masks = [(final_results == id_).astype(np.uint8) for id_ in labels]
  selected_props = []
  best_scores = weighted_scores.max(axis=1)

  # print(partition_function_2.shape,probs_over_objects.shape,final_results.shape,best_scores.shape)

  for idminus1,(mask,score) in enumerate(zip(final_masks,best_scores)):
    seg = encode(np.asfortranarray(mask))
    seg['counts'] = seg['counts'].decode("utf-8")
    prop = dict()
    prop['segmentation'] = seg
    prop['bbox'] = toBbox(seg)
    prop['final_score'] = score
    prop['mask'] = mask
    prop['id'] = idminus1+1 #todo ID here is wrong (at least for ytvos)
    selected_props.append(prop)
  return selected_props

def get_flow(filename):
  with open(filename, 'rb') as f:
    magic = np.fromfile(f, np.float32, count=1)
    if 202021.25 != magic:
      print('Magic number incorrect. Invalid .flo file')
    else:
      w = np.fromfile(f, np.int32, count=1)[0]
      h = np.fromfile(f, np.int32, count=1)[0]
      data = np.fromfile(f, np.float32, count=2 * w * h)
      data2D = np.resize(data, (h, w, 2))
      return data2D

def warp_flow(img, flow, binarize=True):
  h, w = flow.shape[:2]
  flow = -flow
  flow[:, :, 0] += np.arange(w)
  flow[:, :, 1] += np.arange(h)[:, np.newaxis]
  res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
  if binarize:
    res = np.equal(res,1).astype(np.uint8)
  return res

def warp_proposals(proposals,optflow_fn):

  flow = get_flow(optflow_fn)
  masks = [prop['mask'] for prop in proposals]
  ids = [prop['id'] for prop in proposals]
  final_scores = [prop['final_score'] for prop in proposals]
  object_scores = [prop['object_score'] for prop in proposals]
  warped_masks = [warp_flow(mask, flow) for mask in masks]
  warped_props = []
  for i, (f_mask,id) in enumerate(zip(warped_masks,ids)):
    warped_seg = encode(np.asfortranarray(f_mask))
    warped_seg['counts'] = warped_seg['counts'].decode("utf-8")
    warped_prop = dict()
    warped_prop['segmentation'] = warped_seg
    warped_prop['bbox'] = toBbox(warped_seg)
    warped_prop['score'] = 0.5*(final_scores[i]+1)
    # warped_prop['score'] = 0.5 * (copy(object_scores[i]) + 1)
    warped_prop['final_score'] = final_scores[i]
    warped_prop['object_score'] = object_scores[i]
    warped_prop['mask'] = f_mask
    warped_prop['id'] = id
    warped_props.append(warped_prop)
  return warped_props

def update_templates(templates,next_props):
  new_templates = copy(next_props)
  for prop,template in zip(new_templates,templates):
    prop['ReID'] = template['ReID']
    prop['id'] = template['id']
  return new_templates

pascal_colormap = [
    0     ,         0,         0,
    0.5020,         0,         0,
         0,    0.5020,         0,
    0.5020,    0.5020,         0,
         0,         0,    0.5020,
    0.5020,         0,    0.5020,
         0,    0.5020,    0.5020,
    0.5020,    0.5020,    0.5020,
    0.2510,         0,         0,
    0.7529,         0,         0,
    0.2510,    0.5020,         0,
    0.7529,    0.5020,         0,
    0.2510,         0,    0.5020,
    0.7529,         0,    0.5020,
    0.2510,    0.5020,    0.5020,
    0.7529,    0.5020,    0.5020,
         0,    0.2510,         0,
    0.5020,    0.2510,         0,
         0,    0.7529,         0,
    0.5020,    0.7529,         0,
         0,    0.2510,    0.5020,
    0.5020,    0.2510,    0.5020,
         0,    0.7529,    0.5020,
    0.5020,    0.7529,    0.5020,
    0.2510,    0.2510,         0,
    0.7529,    0.2510,         0,
    0.2510,    0.7529,         0,
    0.7529,    0.7529,         0,
    0.2510,    0.2510,    0.5020,
    0.7529,    0.2510,    0.5020,
    0.2510,    0.7529,    0.5020,
    0.7529,    0.7529,    0.5020,
         0,         0,    0.2510,
    0.5020,         0,    0.2510,
         0,    0.5020,    0.2510,
    0.5020,    0.5020,    0.2510,
         0,         0,    0.7529,
    0.5020,         0,    0.7529,
         0,    0.5020,    0.7529,
    0.5020,    0.5020,    0.7529,
    0.2510,         0,    0.2510,
    0.7529,         0,    0.2510,
    0.2510,    0.5020,    0.2510,
    0.7529,    0.5020,    0.2510,
    0.2510,         0,    0.7529,
    0.7529,         0,    0.7529,
    0.2510,    0.5020,    0.7529,
    0.7529,    0.5020,    0.7529,
         0,    0.2510,    0.2510,
    0.5020,    0.2510,    0.2510,
         0,    0.7529,    0.2510,
    0.5020,    0.7529,    0.2510,
         0,    0.2510,    0.7529,
    0.5020,    0.2510,    0.7529,
         0,    0.7529,    0.7529,
    0.5020,    0.7529,    0.7529,
    0.2510,    0.2510,    0.2510,
    0.7529,    0.2510,    0.2510,
    0.2510,    0.7529,    0.2510,
    0.7529,    0.7529,    0.2510,
    0.2510,    0.2510,    0.7529,
    0.7529,    0.2510,    0.7529,
    0.2510,    0.7529,    0.7529,
    0.7529,    0.7529,    0.7529,
    0.1255,         0,         0,
    0.6275,         0,         0,
    0.1255,    0.5020,         0,
    0.6275,    0.5020,         0,
    0.1255,         0,    0.5020,
    0.6275,         0,    0.5020,
    0.1255,    0.5020,    0.5020,
    0.6275,    0.5020,    0.5020,
    0.3765,         0,         0,
    0.8784,         0,         0,
    0.3765,    0.5020,         0,
    0.8784,    0.5020,         0,
    0.3765,         0,    0.5020,
    0.8784,         0,    0.5020,
    0.3765,    0.5020,    0.5020,
    0.8784,    0.5020,    0.5020,
    0.1255,    0.2510,         0,
    0.6275,    0.2510,         0,
    0.1255,    0.7529,         0,
    0.6275,    0.7529,         0,
    0.1255,    0.2510,    0.5020,
    0.6275,    0.2510,    0.5020,
    0.1255,    0.7529,    0.5020,
    0.6275,    0.7529,    0.5020,
    0.3765,    0.2510,         0,
    0.8784,    0.2510,         0,
    0.3765,    0.7529,         0,
    0.8784,    0.7529,         0,
    0.3765,    0.2510,    0.5020,
    0.8784,    0.2510,    0.5020,
    0.3765,    0.7529,    0.5020,
    0.8784,    0.7529,    0.5020,
    0.1255,         0,    0.2510,
    0.6275,         0,    0.2510,
    0.1255,    0.5020,    0.2510,
    0.6275,    0.5020,    0.2510,
    0.1255,         0,    0.7529,
    0.6275,         0,    0.7529,
    0.1255,    0.5020,    0.7529,
    0.6275,    0.5020,    0.7529,
    0.3765,         0,    0.2510,
    0.8784,         0,    0.2510,
    0.3765,    0.5020,    0.2510,
    0.8784,    0.5020,    0.2510,
    0.3765,         0,    0.7529,
    0.8784,         0,    0.7529,
    0.3765,    0.5020,    0.7529,
    0.8784,    0.5020,    0.7529,
    0.1255,    0.2510,    0.2510,
    0.6275,    0.2510,    0.2510,
    0.1255,    0.7529,    0.2510,
    0.6275,    0.7529,    0.2510,
    0.1255,    0.2510,    0.7529,
    0.6275,    0.2510,    0.7529,
    0.1255,    0.7529,    0.7529,
    0.6275,    0.7529,    0.7529,
    0.3765,    0.2510,    0.2510,
    0.8784,    0.2510,    0.2510,
    0.3765,    0.7529,    0.2510,
    0.8784,    0.7529,    0.2510,
    0.3765,    0.2510,    0.7529,
    0.8784,    0.2510,    0.7529,
    0.3765,    0.7529,    0.7529,
    0.8784,    0.7529,    0.7529,
         0,    0.1255,         0,
    0.5020,    0.1255,         0,
         0,    0.6275,         0,
    0.5020,    0.6275,         0,
         0,    0.1255,    0.5020,
    0.5020,    0.1255,    0.5020,
         0,    0.6275,    0.5020,
    0.5020,    0.6275,    0.5020,
    0.2510,    0.1255,         0,
    0.7529,    0.1255,         0,
    0.2510,    0.6275,         0,
    0.7529,    0.6275,         0,
    0.2510,    0.1255,    0.5020,
    0.7529,    0.1255,    0.5020,
    0.2510,    0.6275,    0.5020,
    0.7529,    0.6275,    0.5020,
         0,    0.3765,         0,
    0.5020,    0.3765,         0,
         0,    0.8784,         0,
    0.5020,    0.8784,         0,
         0,    0.3765,    0.5020,
    0.5020,    0.3765,    0.5020,
         0,    0.8784,    0.5020,
    0.5020,    0.8784,    0.5020,
    0.2510,    0.3765,         0,
    0.7529,    0.3765,         0,
    0.2510,    0.8784,         0,
    0.7529,    0.8784,         0,
    0.2510,    0.3765,    0.5020,
    0.7529,    0.3765,    0.5020,
    0.2510,    0.8784,    0.5020,
    0.7529,    0.8784,    0.5020,
         0,    0.1255,    0.2510,
    0.5020,    0.1255,    0.2510,
         0,    0.6275,    0.2510,
    0.5020,    0.6275,    0.2510,
         0,    0.1255,    0.7529,
    0.5020,    0.1255,    0.7529,
         0,    0.6275,    0.7529,
    0.5020,    0.6275,    0.7529,
    0.2510,    0.1255,    0.2510,
    0.7529,    0.1255,    0.2510,
    0.2510,    0.6275,    0.2510,
    0.7529,    0.6275,    0.2510,
    0.2510,    0.1255,    0.7529,
    0.7529,    0.1255,    0.7529,
    0.2510,    0.6275,    0.7529,
    0.7529,    0.6275,    0.7529,
         0,    0.3765,    0.2510,
    0.5020,    0.3765,    0.2510,
         0,    0.8784,    0.2510,
    0.5020,    0.8784,    0.2510,
         0,    0.3765,    0.7529,
    0.5020,    0.3765,    0.7529,
         0,    0.8784,    0.7529,
    0.5020,    0.8784,    0.7529,
    0.2510,    0.3765,    0.2510,
    0.7529,    0.3765,    0.2510,
    0.2510,    0.8784,    0.2510,
    0.7529,    0.8784,    0.2510,
    0.2510,    0.3765,    0.7529,
    0.7529,    0.3765,    0.7529,
    0.2510,    0.8784,    0.7529,
    0.7529,    0.8784,    0.7529,
    0.1255,    0.1255,         0,
    0.6275,    0.1255,         0,
    0.1255,    0.6275,         0,
    0.6275,    0.6275,         0,
    0.1255,    0.1255,    0.5020,
    0.6275,    0.1255,    0.5020,
    0.1255,    0.6275,    0.5020,
    0.6275,    0.6275,    0.5020,
    0.3765,    0.1255,         0,
    0.8784,    0.1255,         0,
    0.3765,    0.6275,         0,
    0.8784,    0.6275,         0,
    0.3765,    0.1255,    0.5020,
    0.8784,    0.1255,    0.5020,
    0.3765,    0.6275,    0.5020,
    0.8784,    0.6275,    0.5020,
    0.1255,    0.3765,         0,
    0.6275,    0.3765,         0,
    0.1255,    0.8784,         0,
    0.6275,    0.8784,         0,
    0.1255,    0.3765,    0.5020,
    0.6275,    0.3765,    0.5020,
    0.1255,    0.8784,    0.5020,
    0.6275,    0.8784,    0.5020,
    0.3765,    0.3765,         0,
    0.8784,    0.3765,         0,
    0.3765,    0.8784,         0,
    0.8784,    0.8784,         0,
    0.3765,    0.3765,    0.5020,
    0.8784,    0.3765,    0.5020,
    0.3765,    0.8784,    0.5020,
    0.8784,    0.8784,    0.5020,
    0.1255,    0.1255,    0.2510,
    0.6275,    0.1255,    0.2510,
    0.1255,    0.6275,    0.2510,
    0.6275,    0.6275,    0.2510,
    0.1255,    0.1255,    0.7529,
    0.6275,    0.1255,    0.7529,
    0.1255,    0.6275,    0.7529,
    0.6275,    0.6275,    0.7529,
    0.3765,    0.1255,    0.2510,
    0.8784,    0.1255,    0.2510,
    0.3765,    0.6275,    0.2510,
    0.8784,    0.6275,    0.2510,
    0.3765,    0.1255,    0.7529,
    0.8784,    0.1255,    0.7529,
    0.3765,    0.6275,    0.7529,
    0.8784,    0.6275,    0.7529,
    0.1255,    0.3765,    0.2510,
    0.6275,    0.3765,    0.2510,
    0.1255,    0.8784,    0.2510,
    0.6275,    0.8784,    0.2510,
    0.1255,    0.3765,    0.7529,
    0.6275,    0.3765,    0.7529,
    0.1255,    0.8784,    0.7529,
    0.6275,    0.8784,    0.7529,
    0.3765,    0.3765,    0.2510,
    0.8784,    0.3765,    0.2510,
    0.3765,    0.8784,    0.2510,
    0.8784,    0.8784,    0.2510,
    0.3765,    0.3765,    0.7529,
    0.8784,    0.3765,    0.7529,
    0.3765,    0.8784,    0.7529,
    0.8784,    0.8784,    0.7529]

def save_with_pascal_colormap(filename, arr):
  colmap = (np.array(pascal_colormap) * 255).round().astype("uint8")
  palimage = Image.new('P', (16, 16))
  palimage.putpalette(colmap)
  im = Image.fromarray(np.squeeze(arr.astype("uint8")))
  im2 = im.quantize(palette=palimage)
  im2.save(filename)

def save_pngs(proposals,output_fn,empty=False):
  png = np.zeros_like(proposals[0]['mask'])
  if not empty:
    for prop in proposals:
      # png[prop['mask'].astype("bool")] = prop['id']+1
      png[prop['mask'].astype("bool")] = prop['id'] #todo check if these id assignments work
  output_fol = '/'.join(output_fn.split('/')[:-1])
  if not os.path.exists(output_fol):
    os.makedirs(output_fol)
  save_with_pascal_colormap(output_fn, png)

def draw_mask(im, mask, alpha=0.5, color=None):
  if color is None:
    color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
  else:
    while color>=127:
      color = color-128
    color = PALETTE_RGB[color+1][::-1]
    # color = pascal_colormap[color*3:color*3+2]
  im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),im * (1 - alpha) + color * alpha, im)
  im = im.astype('uint8')
  return im

def save_jpg(image,masks,output_fn):
  output_fol = '/'.join(output_fn.split('/')[:-1])
  if not os.path.exists(output_fol):
    os.makedirs(output_fol)
  for idx, mask in enumerate(masks):
    image = draw_mask(image, mask, color=idx)
  cv2.imwrite(output_fn, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def c(n):
  r = 1-n
  g = n
  b = 0
  return np.array([r,g,b]).reshape((1,1,3))

def viz_scores(all_scores,weighted_scores,proposals,image_fn,input_images,gt_scores = None):
  out_dir = "DAVIS/viz2/"
  out_fn = image_fn.replace(input_images,out_dir)
  output_fol = '/'.join(out_fn.split('/')[:-1])
  if not os.path.exists(output_fol):
    os.makedirs(output_fol)
  box_size = 100
  # ReID_scores = all_scores[1]
  viz_scores = all_scores.transpose([1,2,0])
  image = imread(image_fn)
  # whole_image = np.zeros((box_size*(ReID_scores.shape[0]+1),box_size*ReID_scores.shape[1],3))
  whole_image = np.zeros((box_size * (viz_scores.shape[0] + 1), box_size * viz_scores.shape[1], 3))

  boxes = [prop['bbox'] for prop in proposals]
  segs = [prop['segmentation'] for prop in proposals]
  for id,(box,seg) in enumerate(zip(boxes,segs)):
    x, y, w, h = box
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    if min(h,w) >= 2:
      crop = image[y:y + h-1, x:x + w-1,:]
      recrop = cv2.resize(crop, dsize=(box_size, box_size), interpolation=cv2.INTER_LINEAR)
      mask = decode(seg)
      mask_crop = mask[y:y + h-1, x:x + w-1]
      mask_recrop = cv2.resize(mask_crop, dsize=(box_size, box_size), interpolation=cv2.INTER_LINEAR)
      recrop_with_mask = draw_mask(recrop,mask_recrop,alpha=0.3,color=1)
    else:
      recrop_with_mask = np.zeros((box_size, box_size,3))
    whole_image[0:box_size,box_size*id:box_size*(id+1),:] = recrop_with_mask
  for i,rr in enumerate(viz_scores):
    for j,gg in enumerate(rr):
      for k,ss in enumerate(gg):
        # crop = np.ones((box_size, box_size, 3)) * ss * 255
        # crop = np.ones((box_size, box_size, 3)) * c(ss) * 255
        # whole_image[box_size*(i+1):box_size*(i+2), box_size * j:box_size * (j + 1), :] = crop
        crop = np.ones((box_size//5, box_size//2, 3)) * c(ss) * 255
        whole_image[box_size*(i+1)+box_size//5*k:box_size*(i+1) + box_size//5*(k+1), box_size*j:box_size*j+box_size//2, :] = crop

      w_score = weighted_scores[i][j]
      crop = np.ones((box_size//2, box_size // 2, 3)) * c(w_score) * 255
      whole_image[box_size*(i+1):box_size*(i+1)+box_size//2,box_size*j+box_size//2:box_size*(j+1),:] = crop

      gt_score = gt_scores[0][i][j]
      crop = np.ones((box_size//2, box_size // 2, 3)) * c(gt_score) * 255
      whole_image[box_size*(i+1)+box_size//2:box_size*(i+2), box_size * j + box_size // 2:box_size * (j + 1), :] = crop

      whole_image[:, box_size*j:box_size*j+1, :] = 0
    whole_image[box_size*(i+1):box_size*(i+1)+1, :, :] = 0

    highest_score = weighted_scores.argmax(axis=1)[i]
    highest_gt = gt_scores[0].argmax(axis=1)[i]
    whole_image[box_size*(i+1)+box_size//8:box_size*(i+1)+box_size//4+box_size//8,
      box_size*highest_score+box_size//2+box_size//8:box_size*highest_score+box_size//2+box_size//4+box_size*1//8,:] = 0
    whole_image[box_size*(i+1)+box_size//2+box_size//8:box_size*(i+1)+box_size//2+box_size//4+box_size//8,
      box_size*highest_gt+box_size//2+box_size//8:box_size*highest_gt+box_size//2+box_size//4+box_size*1//8,:] = 0

  imsave(out_fn,whole_image)

def eval_video(final_solution,input_images,ground_truth_anns):
  _,example_prop = final_solution[0]
  scores = np.zeros(len(example_prop))
  for image_fn,selected_props in final_solution[1:-1]:
    gt_fn = image_fn.replace(input_images, ground_truth_anns).replace('.jpg', '.png')
    gt_props = read_ann(gt_fn)
    segs = [prop['segmentation'] for prop in selected_props]
    gt_segs = [templ['segmentation'] for templ in gt_props]
    gt_ids = [templ['id'] for templ in gt_props]
    for temp_id,seg in enumerate(segs):
      gt_id = temp_id + 1
      if gt_id not in gt_ids:
        if area(seg)==0:
          score = 1
        else:
          score = 0
      else:
        gt_seg = [c_seg for c_seg,c_id in zip(gt_segs,gt_ids) if c_id == gt_id][0]
        score = iou([seg,], [gt_seg,], np.array([0], np.uint8))[0,0]
      scores[temp_id]+=score
  final_scores = scores/(len(final_solution)-2)
  return final_scores

