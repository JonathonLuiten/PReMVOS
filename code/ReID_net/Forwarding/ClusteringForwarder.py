import tensorflow as tf
import numpy
import os
import pickle
import time
import sys
sys.setrecursionlimit(10000)

from .Forwarder import Forwarder
from ReID_net.Log import log
from ReID_net.datasets.Util.Util import username


class ClusteringForwarder(Forwarder):
  def __init__(self, engine):
    super(ClusteringForwarder, self).__init__(engine)
    self.output_folder = "/home/" + username() + "/vision/clustering/data/new_forward/"
    self.RCNN = engine.config.bool("forward_rcnn", False)
    self.network = None
    self.data = None
    self.output_file_name = engine.config.str("output_file_name", "")

  def get_all_latents_from_network(self, y_softmax):
    tag = self.network.tags
    posterior = self.network.inputs_tensors_dict["original_labels"]
    n_total = self.data.num_examples_per_epoch()
    n_processed = 0
    ys = []
    tags = []
    posteriors = []
    while n_processed < n_total:
      start = time.time()
      y_val, tag_val, posterior_val = self.session.run([y_softmax, tag, posterior])
      y_val = y_val[0]
      curr_size = y_val.shape[0]
      for i in range(curr_size):
        ys.append(y_val[i])
        tags.append(tag_val[i])
        posteriors.append(posterior_val[i])
      n_processed += curr_size
      print(n_processed, "/", n_total, " elapsed: ", time.time() - start, file=log.v5)
    self.export_data(ys, tags, posteriors)

  def get_all_latents_from_faster_rcnn(self):
    output_layer = self.network.get_output_layer()
    scores = output_layer.classification_outputs
    features = output_layer.classification_features
    data = self.engine.valid_data
    tags = self.network.inputs_tensors_dict["tags"]
    n_examples = data.num_examples_per_epoch()
    n_processed = 0
    # note that most of these are not used anymore and could be removed
    out_features = []
    out_tags = []
    out_hyps = []
    out_ts = []
    out_scores = []
    out_bbs = []
    out_labels = []
    out_tracklet_filenames = []
    posteriors = []
    assert self.config.str("dataset", "") == "tracklet_detection"
    while n_processed < n_examples:
      start = time.time()
      scores_val, features_val, tags_val = self.engine.session.run([scores, features, tags])
      tag_val = tags_val[0]
      anns = data.filename_to_anns[tag_val]
      assert len(anns) == len(features_val) == scores_val.shape[0]
      for ann, feat, score in zip(anns, features_val, scores_val):
        t = ann["time"]
        hyp_idx = ann["hyp_idx"]
        bbox = ann["bbox"]
        label = ann["category_id"]
        tracklet_filename = ann["tracklet_filename"]
        # print feat.shape, tags_val, hyp_idx, t, score.argmax(), bbox
        out_features.append(feat)

        #!! this works for KITTI, but maybe not for oxford! (TODO: check!)
        seq_str = tag_val.split("/")[-3]
        class_name = ann["classified_category"]
        annotated = ann["annotated_category"]
        annotated = annotated.replace(" ", "_")
        tag_val_fixed = seq_str + "___" + str(hyp_idx) + "___" + str(t) + "___" + class_name + "___" + annotated

        out_tags.append(tag_val_fixed)
        out_hyps.append(hyp_idx)
        out_ts.append(t)
        out_scores.append(score)
        out_bbs.append(bbox)
        out_labels.append(label)
        out_tracklet_filenames.append(tracklet_filename)
        posteriors.append(1)  # dummy for now
      n_processed += 1
      print(n_processed, "/", n_examples, "elapsed:", time.time() - start)

    self.export_data(numpy.array(out_features), out_tags, numpy.array(posteriors))

  def export_data(self, ys, tags, posteriors):
    ### Save vectors and tags
    ys = numpy.array(ys)
    print("Saving pickled output", file=log.v5)
    start = time.time()
    results = {"ys": ys, "tags": tags, "posteriors": posteriors}
    if self.output_file_name == "":
      output_file_name = self.output_folder + str(len(os.listdir(self.output_folder))).zfill(3) + ".pkl"
    else:
      output_file_name = self.output_folder + self.output_file_name
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results, outputfile, pickle.HIGHEST_PROTOCOL)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)
    #return ys, tags, posteriors

  def forward(self, network, data, save_results=True, save_logits=False):
    self.network = network
    self.data = data

    if self.RCNN:
      #output = tf.expand_dims(self.network.get_output_layer().classification_features, axis=0)
      self.get_all_latents_from_faster_rcnn()
    else:
      output = self.network.get_output_layer().outputs
      self.get_all_latents_from_network(output)
