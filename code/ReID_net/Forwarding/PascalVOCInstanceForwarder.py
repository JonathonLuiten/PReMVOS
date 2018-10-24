import pickle
from scipy.misc import imsave
from ReID_net.Log import log

import numpy
import tensorflow as tf

from ReID_net.Forwarding.Forwarder import ImageForwarder
from ReID_net.Measures import compute_measures_for_binary_segmentation, average_measures


class PascalVOCInstanceForwarder(ImageForwarder):

  # return the generated mask, which can be used by the click tool
  def forward(self, network, data, save_results=True, save_logits=False):
    n_total = data.num_examples_per_epoch()
    n_processed = 0
    targets = network.raw_labels
    ys = network.y_softmax

    # e.g. used for resizing
    ys = self._adjust_results_to_targets(ys, targets)

    measures = []
    ys_argmax_values = []
    while n_processed < n_total:
      n, new_measures, ys_argmax_val, _, _ = self._process_forward_minibatch(data, network, save_logits, save_results,
                                                                 targets, ys, n_processed)
      measures += new_measures
      # ys_argmax_values += ys_argmax_val
      n_processed += n
      print(n_processed, "/", n_total, file=log.v5)
    if self.ignore_first_and_last_results:
      measures = measures[1:-1]
    elif self.ignore_first_result:
      measures = measures[1:]

    measures = average_measures(measures)
    if hasattr(data, "video_tag"):
      video_idx = data.get_video_idx()
      print("sequence", video_idx + 1, data.video_tag(video_idx), measures, file=log.v1)
    else:
      print(measures, file=log.v1)

    return ys_argmax_val[0, :, :, 0]

  def _process_forward_result(self, y_argmax, logit, target, tag, extraction_vals, main_folder, save_results):
    #hack for avoiding storing logits for frames, which are not evaluated
    if "DO_NOT_STORE_LOGITS" in tag:
      logit = None
      tag = tag.replace("_DO_NOT_STORE_LOGITS", "")

    folder = main_folder + tag.split("/")[-2] + "/"
    tf.gfile.MakeDirs(folder)
    inst = tag.split(":")[-1]
    out_fn = folder + tag.split("/")[-1].split(":")[-2].replace(".jpg", "_" + inst + ".png").replace(".bin", ".png")
    out_fn_logits = out_fn.replace(".png", ".pickle")
    
    target_fn = out_fn.replace(".png", "_target.png")
    measures = compute_measures_for_binary_segmentation(y_argmax, target)
    if save_results:
      y_scaled = (y_argmax ).astype("uint8")
      print(out_fn)
      imsave(out_fn, numpy.squeeze(y_scaled, axis=2))
      # imsave(target_fn, numpy.squeeze(target_scaled, axis=2 ))
    if logit is not None:
      pickle.dump(logit, open(out_fn_logits, "w"), pickle.HIGHEST_PROTOCOL)
    for e in extraction_vals:
      assert e.shape[0] == 1  # batchs size should be 1 here for now
    for name, val in zip(self.extractions, extraction_vals):
      val = val[0]  # remove batch dimension
      sp = out_fn.replace(".png", ".bin").split("/")
      sp[-1] = name + "_" + sp[-1]
      out_fn_extract = "/".join(sp)
      print(out_fn_extract)
      val.tofile(out_fn_extract)
    return measures
