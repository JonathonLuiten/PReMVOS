from .Similarity import SimilarityDataset
import glob
import json
from pycocotools.mask import toBbox

class DAVISForwardSimilarityDataset(SimilarityDataset):
  def __init__(self, config, subset, coord):

    # old_proposal_directory = "/home/luiten/vision/PReMVOS/first_frame_no_ReID/%s/"
    old_proposal_directory = config.str("bb_input_dir", None)
    data_directory = config.str("image_input_dir", None)

    # old_proposal_directory = "/home/luiten/vision/PReMVOS/proposals_with_flow/%s/"
    # old_proposal_directory = "/home/luiten/vision/PReMVOS/post_proposal_expansion_json_with_flow/%s/"
    # sets = ['test-challenge/', 'val/', 'test-dev/']
    # sets = ['val/',]
    annotations = []

    # Read in all proposals
    # for set_id, set in enumerate(sets):
    # folders = sorted(glob.glob(old_proposal_directory.split('%s')[0] + set + '*/'))
    folders = sorted(glob.glob(old_proposal_directory + '*/'))
    for folder in folders:
      seq = folder.split('/')[-2]
      # name = set + seq
      name = seq
      # files = sorted(glob.glob(old_proposal_directory % name + "*.json"))
      files = sorted(glob.glob(old_proposal_directory + name + "/*.json"))
      for file in files:
        timestep = file.split('/')[-1].split('.json')[0]
        with open(file, "r") as f:
          proposals = json.load(f)
        for prop_id, proposal in enumerate(proposals):
          # img_file = "/home/luiten/vision/PReMVOS/home_data/"+name+"/images/"+timestep+".jpg"
          img_file = data_directory + name + "/" + timestep + ".jpg"
          catagory_id = 1
          tag = name+'/'+timestep+'___'+str(prop_id)
          segmentation = proposal["segmentation"]
          bbox = toBbox(segmentation)
          ann = {"img_file":img_file,"category_id":catagory_id,"bbox":bbox,"tag":tag}

          if bbox[2] <= 0 or bbox[3] <= 0:
            continue

          annotations.append(ann)

    super(DAVISForwardSimilarityDataset, self).__init__(config, subset, coord, annotations, n_train_ids=1)
