import glob
from numpy.linalg import norm
import numpy as np
from copy import deepcopy as copy

from MergeTrack.merge_functions import read_ann,read_props
from MergeTrack.ReID_net_functions import ReID_net_init, add_ReID

input_images = "DAVIS/val17/"
input_proposals = "DAVIS/ReID_props/"
first_frame_anns = "DAVIS/val17-ff/"
output_images = "DAVIS/final_results/"
output_proposals = "DAVIS/final_props/"

ReID_net = ReID_net_init()

dataset_max_distances = []
for video_fn in sorted(glob.glob(input_images+"*/")):
  video_proposals = []
  templates = []
  for image_fn in sorted(glob.glob(video_fn+"*")):
    ann_fn = image_fn.replace(input_images,first_frame_anns).replace('.jpg','.png')
    if glob.glob(ann_fn):
      new_templates = read_ann(ann_fn)
      new_templates = add_ReID(new_templates, image_fn, ReID_net)

      # import json
      # ff_fn = image_fn.replace(input_images, "DAVIS/ff_test/").replace('.jpg', '.json')
      # with open(ff_fn, "r") as f:
      #   new_templates = json.load(f)
      # for id, templ in enumerate(new_templates):
      #   templ['ReID'] = np.array(templ['ReID'])
      #   templ['id'] = id

      templates = templates + new_templates
    prop_fn = image_fn.replace(input_images,input_proposals).replace('.jpg','.json')
    proposals = read_props(prop_fn)
    video_proposals.append(proposals)

  ReIDs = [[prop['ReID'] for prop in props] for props in video_proposals]
  template_ReIDs = [templ['ReID'] for templ in templates]
  all_reid_distances = [np.array([[norm(c_reid - gt_reid) for c_reid in curr] for gt_reid in template_ReIDs]) for curr in ReIDs]
  all_reid_distances_no_inf = copy(all_reid_distances)

  for mat in all_reid_distances_no_inf:
    mat[np.isinf(mat)] = 0

  max_distances = np.array([mat.max(axis=1) if mat.shape[1]>0 else np.zeros((mat.shape[0])) for mat in all_reid_distances_no_inf]).max(axis=0)
  print(max_distances)
  dataset_max_distances.append(max_distances.max())

print(np.array(dataset_max_distances).max())