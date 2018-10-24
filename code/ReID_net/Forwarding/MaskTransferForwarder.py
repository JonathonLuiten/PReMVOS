import tensorflow as tf

import ReID_net.Measures as Measures

import ReID_net.Util as Util
from .Forwarder import Forwarder, merge_multi_samples
from .OneshotForwarder import OneshotForwarder
from ReID_net.Measures import compute_iou_for_binary_segmentation
from ReID_net.datasets.Util.MaskDamager import damage_mask
from ReID_net.Log import log
import numpy
import pickle
from scipy.misc import imsave


class MaskTransferOneshotForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(MaskTransferOneshotForwarder, self).__init__(engine)
    self.base_forwarder = MaskTransferForwarder(engine)
    self.old_mask_scale_factor = self.config.float("old_mask_scale_factor", 0.0)
    self.old_mask_shift_factor = self.config.float("old_mask_shift_factor", 0.0)
    self.old_mask_shift_absolute = self.config.float("old_mask_shift_absolute", 0.0)

  def _base_forward(self, network, data, save_results, save_logits):
    video_idx = data.get_video_idx()
    self.base_forwarder.forward(network, data, save_results, save_logits, video_idx)

  def _finetune(self, video_idx, n_finetune_steps, start_step=0):
    frame_idx = 0
    tag = self.train_data.video_tag(video_idx)
    self.train_data.set_video_idx(video_idx)

    #problem: for frame 0 there is no old mask
    #for now: just use the current mask as input and optionally damage it
    #also: the backward flow is set to zero here, since there is no preceding frame
    #maybe: generate some artificial backward flow
    old_mask_original = self.train_data.label_for_video_frame(frame_idx=0)
    for idx in range(start_step, start_step + n_finetune_steps):
      old_mask_damaged = damage_mask(old_mask_original, self.old_mask_scale_factor, self.old_mask_shift_absolute,
                                     self.old_mask_shift_factor)
      feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True, old_mask=old_mask_damaged)
      loss, measures, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict)
      loss /= n_imgs
      # TODO: Replace [0] with ingore classes.
      iou = Measures.calc_iou(measures, n_imgs, [0])
      print("finetune on", tag, idx, "/", start_step + n_finetune_steps, "loss:", loss, " iou:", iou, file=log.v5)


class MaskTransferForwarder(Forwarder):
  def __init__(self, engine):
    super(MaskTransferForwarder, self).__init__(engine)
    self.n_test_samples = engine.config.int("n_test_samples", 1)

  def forward(self, network, data, save_results=True, save_logits=False, video_idx=None):
    video_ious = []
    if video_idx is None:
      video_indices = list(range(data.n_videos()))
    else:
      video_indices = [video_idx]
    for video_idx in video_indices:
      data.set_video_idx(video_idx)
      old_mask = data.label_for_video_frame(frame_idx=0)
      n_frames = data.num_examples_per_epoch()
      ious = []
      for frame in range(1, n_frames - 1):
        feed_dict = data.feed_dict_for_video_frame(frame, with_annotations=True, old_mask=old_mask)

        ys = network.y_softmax
        targets = network.raw_labels
        tags = network.tags
        # scale it up!
        ys = tf.image.resize_images(ys, tf.shape(targets)[1:3])
        ys_argmax = tf.expand_dims(tf.arg_max(ys, 3), axis=3)

        if self.n_test_samples == 1:
          logits, new_masks, targets_val, tags_val = self.session.run([ys, ys_argmax, targets, tags], feed_dict)
          assert len(new_masks) == len(targets_val) == len(tags_val) == 1
          new_mask = new_masks[0]
          target = targets_val[0]
          logit = logits[0]
          tag = tags_val[0]
        else:
          idx_imgs = network.index_imgs
          ys_val, idx_imgs_val, targets_val, tags_val = self.session.run([ys, idx_imgs, targets, tags], feed_dict)
          tag = tags_val[0]
          #new_mask = merge_multi_samples(ys_val, idx_imgs_val, targets_val)[1][0]
          logits, new_masks = merge_multi_samples(ys_val, idx_imgs_val, targets_val)
          new_mask = new_masks[0]
          logit = logits[0]
          target = targets_val[0]
          #targets should be raw labels, all should be the same here
          assert all((t == target).all() for t in targets_val[1:])
        iou = compute_iou_for_binary_segmentation(new_mask, target)
        ious.append(iou)

        #save
        #TODO: avoid code duplication from Forwarder.py
        main_folder = "forwarded/" + self.model + "/" + data.subset + "/"
        folder = main_folder + tag.split("/")[-2] + "/"
        tf.gfile.MakeDirs(folder)
        out_fn = folder + tag.split("/")[-1].replace(".jpg", ".png")
        if save_results:
          y_scaled = numpy.squeeze((new_mask * 255).astype("uint8"), axis=2)
          imsave(out_fn, y_scaled)
        if save_logits:
          out_fn_logits = out_fn.replace(".png", ".pickle")
          pickle.dump(logit, open(out_fn_logits, "w"), pickle.HIGHEST_PROTOCOL)

        print(frame, "/", (n_frames - 2), out_fn, iou, file=log.v5)

        old_mask = new_mask
      video_iou = numpy.mean(ious)
      video_ious.append(video_iou)
      print("sequence", video_idx + 1, data.video_tag(video_idx), video_iou, file=log.v1)
    if video_idx is None:
      print("mean", numpy.mean(video_ious), file=log.v1)
