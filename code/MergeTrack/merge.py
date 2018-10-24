#!/usr/bin/env python

# Psudo code:
'''
Create empty set of objects to track
Create current frame propoals
Create next frame proposals, for both next and current frame

for each image:
  Check if new first frame annos avaliable
  for each new first frame anno object:
    Read in anno
    Convert to json format, with bb and mask
    Calculate ReID embedding and add to JSON
    Add first frame anno as final solution
    Add first frame anno into next info for current frame
    Assign score to 1
    Calculate warped first frame anno, mask, bb and ReID
    Add to list of proposals for next frame, and next_info in objects to track
  Curr frame proposals = next frame proposals from last frame
  Read in new curr frame propals and add to curr frame proposals list
  for each not first frame anno object:
    Calculate weightings for WarpScores based on previous Total score for the assingned track from previous
    for each curr frame proposal:
      Calculate ReID score to each track next_info
      Calculate InVReID score to each track next_info
      Calculate WarpScore to each track next_info
      Calculate InvWarpScore to each track next_info
      Calculate Total Score for each proposal to each track
    Choose proposals for each track with argmax
    # maybe: choose blank proposal if score is too low
  Overlay proposals on one image choosing proposals with highest score
  From new overlaid proposals calculate bounding boxes
  Calculate new refined mask from new bounding boxes
  Overlay new refined masks again, again choosing that with the best score
  Add these to final output results
  Calculate warped mask into next frame, extract bb
  Calculate new refined warped mask, and ReID
  Add to list of proposals for next frame with current ReID
  Add to next info in objects to track with first frame ReID (explicit copy proabably needed)
'''

print("Starting up")
import glob
import numpy as np
import os
from copy import deepcopy as copy
from scipy.misc import imread

from MergeTrack.merge_functions import read_ann,read_props,calculate_scores,calculate_selected_props,\
  remove_mask_overlap,warp_proposals,update_templates,save_pngs,viz_scores,eval_video,calculate_gt_scores,\
  probabilitic_combination

from MergeTrack.refinement_net_functions import refinement_net_init, do_refinement
refinement_net = refinement_net_init()

from MergeTrack.ReID_net_functions import ReID_net_init, add_ReID
ReID_net = ReID_net_init()

input_images = "../data/DAVIS/JPEGImages/480p/"
first_frame_anns = "../data/DAVIS/Annotations/480p/"
input_proposals = "../output/intermediate/ReID_proposals/"
input_optical_flow = "../output/intermediate/flow/"
output_images = "../output/final/"

curr_run_num = 0
total_to_run = 1

def do_video(video_fn):

  print("Starting", video_fn)
  final_solution = []
  templates = []
  next_props = []
  image_fn_list = sorted(glob.glob(video_fn + "*"))
  for image_id, image_fn in enumerate(image_fn_list):
    ann_fn = image_fn.replace(input_images, first_frame_anns).replace('.jpg', '.png')
    if glob.glob(ann_fn) and "00000.jpg" in image_fn:
      new_templates = read_ann(ann_fn)
      new_templates = add_ReID(new_templates, image_fn, ReID_net)
      templates = templates + copy(new_templates)
      next_props = next_props + copy(new_templates)
    if templates:
      prop_fn = image_fn.replace(input_images, input_proposals).replace('.jpg', '.json')
      proposals = next_props + read_props(prop_fn)

      all_scores = calculate_scores(proposals, templates)

      weighted_scores = np.dot(normalised_weights, all_scores.transpose((1, 0, 2)), )
      object_scores = np.dot(np.array([1, 1]), all_scores[:2, :, :].transpose((1, 0, 2)), )

      selected_props = calculate_selected_props(proposals, weighted_scores, templates, score_thesh, object_scores)
      selected_props = remove_mask_overlap(selected_props)

      optflow_fn = image_fn.replace(input_images, input_optical_flow).replace('.jpg', '.flo')
      if glob.glob(optflow_fn):
        next_image_fn = image_fn_list[image_id + 1]
        next_props = warp_proposals(selected_props, optflow_fn)

        next_props = do_refinement(next_props, next_image_fn, refinement_net)
        next_props = add_ReID(next_props, next_image_fn, ReID_net)
        templates = update_templates(templates, next_props)
      else:
        print("END")
      final_solution.append((image_fn, selected_props))
      output_image_fn = image_fn.replace(input_images, output_images).replace('.jpg', '.png')
      print(output_image_fn)
      save_pngs(selected_props, output_image_fn)
    else:
      output_image_fn = image_fn.replace(input_images, output_images).replace('.jpg', '.png')
      print(output_image_fn)
      img = imread(image_fn)
      temp_props = [dict()]
      temp_props[0]['mask'] = np.zeros_like(img[:, :, 1]).astype(np.uint8)
      save_pngs(temp_props, output_image_fn, empty=True)

score_thesh = 1e-10

weights = np.array([0.25920137, 0.22541801, 0.0775609,  0.12509281, 0.3127269 ])

normalised_weights = weights / np.sum(weights)

final_scores = []
video_list = sorted(glob.glob(input_proposals+"*/"))
video_list = [v.replace(input_proposals,input_images) for v in video_list]
num_per = int(np.ceil(len(video_list)/total_to_run))
print(curr_run_num,num_per,curr_run_num*num_per,(curr_run_num+1)*num_per)
for video_fn in video_list[curr_run_num*num_per : (curr_run_num+1)*num_per]:
  do_video(video_fn)


