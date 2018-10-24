import pickle
import time
from math import ceil
from scipy.misc import imsave

import numpy
import tensorflow as tf

import ReID_net.Measures as Measures
from ReID_net.Forwarding.OneshotForwarder import OneshotForwarder
from ReID_net.Log import log


class IterativeImageForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(IterativeImageForwarder, self).__init__(engine=engine)
    self.training_rounds = engine.config.int("training_rounds", 5)
    self.round = 0
    self.logits = []

  def _oneshot_forward_video(self, video_idx, save_logits):
    # TODO: The weights are reset for each video sequence in the forward method. Check if this is necessary.
    forward_interval = self.forward_interval
    if forward_interval > self.n_finetune_steps:
      forward_interval = self.n_finetune_steps
      n_partitions = 1
    else:
      n_partitions = int(ceil(float(self.n_finetune_steps) / forward_interval))

    self.train_data.set_video_idx(video_idx)
    # if hasattr(self.train_data, "change_label_to_bbox"):
    #   self.train_data.change_label_to_bbox()

    ys_argmax_values, self.logits = self._base_forward(self.engine.test_network, self.val_data, False, save_logits)
    for self.round in range(self.training_rounds):
      # self.train_data.set_video_annotation(logits)
      for i in range(n_partitions):
        start = time.time()
        self._finetune(video_idx, n_finetune_steps=min(forward_interval, self.n_finetune_steps),
                       start_step=forward_interval * i)
        save_results = self.save_oneshot and i == n_partitions - 1
        save_logits_here = save_logits and i == n_partitions - 1
        ys_argmax_values, self.logits = self._base_forward(self.engine.test_network, self.val_data,
                                                           save_results=save_results,
                                                           save_logits=save_logits_here)
        end = time.time()
        elapsed = end - start
        print("steps:", forward_interval * (i + 1), "elapsed", elapsed, file=log.v4)

  def _base_forward(self, network, data, save_results, save_logits):
    n_total = data.num_examples_per_epoch()
    n_processed = 0
    targets = network.raw_labels
    ys = network.y_softmax

    # e.g. used for resizing
    ys = self._adjust_results_to_targets(ys, targets)

    measures = []
    ys_argmax_values = []
    logits = []
    while n_processed < n_total:
      n, new_measures, ys_argmax, logit, _ = self._process_forward_minibatch(data, network, save_logits, save_results,
                                                                 targets, ys, n_processed)
      measures += new_measures
      ys_argmax_values += list(ys_argmax)
      logits += list(logit)
      n_processed += n
      print(n_processed, "/", n_total, file=log.v5)
    if self.ignore_first_and_last_results:
      measures = measures[1:-1]
    elif self.ignore_first_result:
      measures = measures[1:]

    measures = Measures.average_measures(measures)
    if hasattr(data, "video_tag"):
      video_idx = data.get_video_idx()
      print("sequence", video_idx + 1, data.video_tag(video_idx), measures, file=log.v1)
    else:
      print(measures, file=log.v1)

    return ys_argmax_values, logits

  def _finetune(self, video_idx, n_finetune_steps, start_step=0):
    frame_id = 0
    tag = self.train_data.video_tag(video_idx)
    self.train_data.set_video_idx(video_idx)
    for idx in range(start_step, start_step + n_finetune_steps):
      # for frame_id in xrange(0, len(self.train_data._get_video_data())):
      if self.lucid_interval != -1 and idx % self.lucid_interval == 0:
        print("lucid example", file=log.v5)
        feed_dict = self.train_data.get_lucid_feed_dict()
        loss_scale = self.lucid_loss_scale
      else:
        # feed_dict = self.train_data.feed_dict_for_video_frame(frame_id, with_annotations=True,
        #                                                       old_mask=self.logits[frame_id])
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_id, with_annotations=True)
        loss_scale = 1.0

        loss, measures, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale)
        loss /= n_imgs
        iou = Measures.calc_iou(measures, n_imgs, [0])
        print("finetune on", tag, idx, "/", start_step + n_finetune_steps, "loss:", loss, " iou:", iou, file=log.v5)

  def _process_forward_result(self, y_argmax, logit, target, tag, extraction_vals, main_folder, save_results):
    # hack for avoiding storing logits for frames, which are not evaluated
    if "DO_NOT_STORE_LOGITS" in tag:
      logit = None
      tag = tag.replace("_DO_NOT_STORE_LOGITS", "")

    folder = main_folder + tag.split("/")[-2] + "/"
    tf.gfile.MakeDirs(folder)
    if self.training_rounds > 1:
      out_fn = folder + tag.split("/")[-1].replace(".jpg", "_" + repr(self.round) + ".png").replace(".bin", ".png")
    else:
      out_fn = folder + tag.split("/")[-1].replace(".jpg", "_" + repr(self.round) + ".png").replace(".bin", ".png")
    out_fn_logits = out_fn.replace(".png", ".pickle")

    target_fn = out_fn.replace(".png", "_target.png")
    measures = Measures.compute_measures_for_binary_segmentation(y_argmax, target)
    if save_results:
      y_scaled = (y_argmax).astype("uint8")
      print(out_fn)
      imsave(out_fn, numpy.squeeze(y_scaled * 255, axis=2))
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
