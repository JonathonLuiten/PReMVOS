from ReID_net.Forwarding.OneshotForwarder import OneshotForwarder
from ReID_net.datasets.Util.Timer import Timer
from ReID_net.Measures import average_measures
from ReID_net.Log import log

import numpy
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion

VOID_LABEL = 255


class OnlineAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(OnlineAdaptingForwarder, self).__init__(engine)
    self.n_adaptation_steps = self.config.int("n_adaptation_steps", 12)
    self.adaptation_interval = self.config.int("adaptation_interval", 4)
    self.adaptation_learning_rate = self.config.float("adaptation_learning_rate")
    self.posterior_positive_threshold = self.config.float("posterior_positive_threshold", 0.97)
    self.distance_negative_threshold = self.config.float("distance_negative_threshold", 150.0)
    self.adaptation_loss_scale = self.config.float("adaptation_loss_scale", 0.1)
    self.debug = self.config.bool("adapt_debug", False)
    self.erosion_size = self.config.int("adaptation_erosion_size", 20)
    self.use_positives = self.config.bool("use_positives", True)
    self.use_negatives = self.config.bool("use_negatives", True)

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():
      # finetune on first frame
      self._finetune(video_idx, n_finetune_steps=self.n_finetune_steps)

      network = self.engine.test_network
      targets = network.raw_labels
      ys = network.y_softmax
      ys = self._adjust_results_to_targets(ys, targets)
      data = self.val_data

      n, measures, ys_argmax_val, logits_val, targets_val = self._process_forward_minibatch(
        data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)
      assert n == 1
      n_frames = data.num_examples_per_epoch()

      measures_video = []

      last_mask = targets_val[0]

      for t in range(1, n_frames):
        def get_posteriors():
          n_, _, _, logits_val_, _ = self._process_forward_minibatch(
              data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
          assert n_ == 1
          return logits_val_[0]

        # online adaptation to current frame
        negatives = self._adapt(video_idx, t, last_mask, get_posteriors)

        # forward current frame using adapted model
        n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
            data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
        assert n == 1
        assert len(measures) == 1
        measure = measures[0]
        print("frame", t, ":", measure, file=log.v5)
        measures_video.append(measure)
        last_mask = ys_argmax_val[0]

        # prune negatives from last mask
        # negatives are None if we think that the target is lost
        if negatives is not None and self.use_negatives:
          last_mask[negatives] = 0

      measures_video[:-1] = measures_video[:-1]
      measures_video = average_measures(measures_video)
      print("sequence", video_idx + 1, data.video_tag(video_idx), measures_video, file=log.v1)

  def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn):
    eroded_mask = grey_erosion(last_mask, size=(self.erosion_size, self.erosion_size, 1))
    dt = distance_transform_edt(numpy.logical_not(eroded_mask))

    adaptation_target = numpy.zeros_like(last_mask)
    adaptation_target[:] = VOID_LABEL

    current_posteriors = get_posteriors_fn()
    positives = current_posteriors[:, :, 1] > self.posterior_positive_threshold
    if self.use_positives:
      adaptation_target[positives] = 1

    threshold = self.distance_negative_threshold
    negatives = dt > threshold
    if self.use_negatives:
      adaptation_target[negatives] = 0

    do_adaptation = eroded_mask.sum() > 0

    if self.debug:
      adaptation_target_visualization = adaptation_target.copy()
      adaptation_target_visualization[adaptation_target == 1] = 128
      if not do_adaptation:
        adaptation_target_visualization[:] = VOID_LABEL
      from scipy.misc import imsave
      folder = self.val_data.video_tag().replace("__", "/")
      imsave("forwarded/" + self.model + "/valid/" + folder + "/adaptation_%05d.png" % frame_idx,
             numpy.squeeze(adaptation_target_visualization))

    self.train_data.set_video_idx(video_idx)

    for idx in range(self.n_adaptation_steps):
      do_step = True
      if idx % self.adaptation_interval == 0:
        if do_adaptation:
          feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
          feed_dict[self.train_data.get_label_placeholder()] = adaptation_target
          loss_scale = self.adaptation_loss_scale
          adaption_frame_idx = frame_idx
        else:
          print("skipping current frame adaptation, since the target seems to be lost", file=log.v4)
          do_step = False
      else:
        # mix in first frame to avoid drift
        # (do this even if we think the target is lost, since then this can help to find back the target)
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx=0, with_annotations=True)
        loss_scale = 1.0
        adaption_frame_idx = 0

      if do_step:
        loss, _, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale,
                                                  learning_rate=self.adaptation_learning_rate)
        assert n_imgs == 1
        print("adapting on frame", adaption_frame_idx, "of sequence", video_idx + 1, \
            self.train_data.video_tag(video_idx), "loss:", loss, file=log.v4)
    if do_adaptation:
      return negatives
    else:
      return None
