import numpy as np
import os
import time
from .Forwarder import Forwarder
from ReID_net.Log import log
import glob
import json

class ReIDForwarder(Forwarder):
  def __init__(self, engine):
    super(ReIDForwarder, self).__init__(engine)
    self.network = None
    self.data = None

  def get_all_latents_from_network(self, y_softmax):
    tag = self.network.tags
    n_total = self.data.num_examples_per_epoch()
    n_processed = 0
    ys = []
    tags = []
    while n_processed < n_total:
      start = time.time()
      y_val, tag_val = self.session.run([y_softmax, tag])
      y_val = y_val[0]
      curr_size = y_val.shape[0]
      for i in range(curr_size):
        ys.append(y_val[i])
        tags.append(tag_val[i].decode('utf-8'))
      n_processed += curr_size
      print(n_processed, "/", n_total, " elapsed: ", time.time() - start)
    self.export_data(ys, tags)

  def export_data(self, ys, tags):
    print("EXPORTING DATA")

    # old_proposal_directory = "/home/luiten/vision/PReMVOS/first_frame_no_ReID/%s/"
    # new_proposal_dir = "/home/luiten/vision/PReMVOS/first_frame_final/%s/"

    old_proposal_directory = self.config.str("bb_input_dir", None)
    new_proposal_dir = self.config.str("output_dir", None)

    # old_proposal_directory = "/home/luiten/vision/PReMVOS/proposals_with_flow/%s/"
    # new_proposal_dir = "/home/luiten/vision/PReMVOS/final_proposals/%s/"

    # old_proposal_directory = "/home/luiten/vision/PReMVOS/post_proposal_expansion_json_with_flow/%s/"
    # new_proposal_dir = "/home/luiten/vision/PReMVOS/expanded_final_props/%s/"
    # sets = ['test-challenge/', 'val/', 'test-dev/']
    # sets = ['val/',]
    all_proposals = dict()

    # Read in all proposals
    # for set_id, set in enumerate(sets):
    folders = sorted(glob.glob(old_proposal_directory + '*/'))
    for folder in folders:
      seq = folder.split('/')[-2]
      name = seq
      files = sorted(glob.glob(old_proposal_directory + name + "/*.json"))
      for file in files:
        timestep = file.split('/')[-1].split('.json')[0]

        # Get proposals:
        with open(file, "r") as f:
          proposals = json.load(f)
        all_proposals[name+'/'+timestep] = proposals

    print("READ IN ALL PROPOSALS")

    # Insert embeddings into proposals
    for y,tag in zip(ys,tags):
      nametime,prop_id = tag.split('___')
      prop_id = int(prop_id)
      y = np.array(y).tolist()
      all_proposals[nametime][prop_id]["ReID"] = y

    print("INSERTED EMBEDDINGS")

    # Save out to file
    # for set_id, set in enumerate(sets):
    folders = sorted(glob.glob(old_proposal_directory + '*/'))
    for folder in folders:
      seq = folder.split('/')[-2]
      name = seq
      files = sorted(glob.glob(old_proposal_directory + name + "/*.json"))
      for file in files:
        timestep = file.split('/')[-1].split('.json')[0]

        # Save new proposals:
        new_file = new_proposal_dir + name + "/" + timestep + ".json"
        if not os.path.exists(new_proposal_dir + name):
          os.makedirs(new_proposal_dir + name)
        with open(new_file, 'w') as f:
          json.dump(all_proposals[name+'/'+timestep], f)

  def forward(self, network, data, save_results=True, save_logits=False):
    self.network = network
    self.data = data

    output = self.network.get_output_layer().outputs
    self.get_all_latents_from_network(output)
