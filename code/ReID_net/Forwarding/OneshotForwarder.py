import ReID_net.Measures as Measures
import ReID_net.Util as Util
from ReID_net.Forwarding import InteractiveImageForwarder
from ReID_net.Forwarding.Forwarder import ImageForwarder
import time
from math import ceil
from ReID_net.Log import log


class OneshotForwarder(ImageForwarder):
  def __init__(self, engine):
    super(OneshotForwarder, self).__init__(engine)
    self.val_data = self.engine.valid_data
    if hasattr(self.engine, "train_data"):
      self.train_data = self.engine.train_data
    self.trainer = self.engine.trainer
    self.forward_interval = self.config.int("forward_interval", 9999999)
    self.forward_initial = self.config.bool("forward_initial", False)
    self.n_finetune_steps = self.config.int("n_finetune_steps", 40)
    self.video_range = self.config.int_list("video_range", [])
    self.video_ids = self.config.int_list("video_ids", [])
    assert len(self.video_range) == 0 or len(self.video_ids) == 0, "cannot specify both"
    self.save_oneshot = self.config.bool("save_oneshot", False)
    self.lucid_interval = self.config.int("lucid_interval", -1)
    self.lucid_loss_scale = self.config.float("lucid_loss_scale", 1.0)

  def _maybe_adjust_output_layer_for_multiple_objects(self):
    if not self.config.bool("adjustable_output_layer", False):
      return
    assert hasattr(self.val_data, "get_number_of_objects_for_video")
    n_objects = self.val_data.get_number_of_objects_for_video()
    print("adjusting output layer for", n_objects, "objects", file=log.v3)
    self.engine.train_network.get_output_layer().adjust_weights_for_multiple_objects(self.session, n_objects)

  def forward(self, network, data, save_results=True, save_logits=False):
    if len(self.video_range) != 0:
      video_ids = list(range(self.video_range[0], self.video_range[1]))
    elif len(self.video_ids) != 0:
      video_ids = self.video_ids
    else:
      video_ids = list(range(0, self.train_data.n_videos()))
    for video_idx in video_ids:
      tag = self.train_data.video_tag(video_idx)
      print("finetuning on", tag, file=log.v4)

      # reset weights and optimizer for next video
      self.engine.try_load_weights()
      self.engine.reset_optimizer()
      self.val_data.set_video_idx(video_idx)
      self._maybe_adjust_output_layer_for_multiple_objects()

      print("steps:", 0, file=log.v4)
      self._oneshot_forward_video(video_idx, save_logits)

  def _oneshot_forward_video(self, video_idx, save_logits):
    forward_interval = self.forward_interval
    if forward_interval > self.n_finetune_steps:
      forward_interval = self.n_finetune_steps
      n_partitions = 1
    else:
      n_partitions = int(ceil(float(self.n_finetune_steps) / forward_interval))
    if self.forward_initial:
      self._base_forward(self.engine.test_network, self.val_data, save_results=False, save_logits=False)
    for i in range(n_partitions):
      start = time.time()
      self._finetune(video_idx, n_finetune_steps=min(forward_interval, self.n_finetune_steps),
                     start_step=forward_interval * i)
      save_results = self.save_oneshot and i == n_partitions - 1
      save_logits_here = save_logits and i == n_partitions - 1
      self._base_forward(self.engine.test_network, self.val_data, save_results=save_results,
                         save_logits=save_logits_here)
      end = time.time()
      elapsed = end - start
      print("steps:", forward_interval * (i + 1), "elapsed", elapsed, file=log.v4)

  def _base_forward(self, network, data, save_results, save_logits):
    super(OneshotForwarder, self).forward(self.engine.test_network, self.val_data, save_results, save_logits)

  def _finetune(self, video_idx, n_finetune_steps, start_step=0):
    frame_idx = 0
    tag = self.train_data.video_tag(video_idx)
    self.train_data.set_video_idx(video_idx)
    for idx in range(start_step, start_step + n_finetune_steps):
      if self.lucid_interval != -1 and idx % self.lucid_interval == 0:
        print("lucid example", file=log.v5)
        feed_dict = self.train_data.get_lucid_feed_dict()
        loss_scale = self.lucid_loss_scale
      else:
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
        loss_scale = 1.0
      loss, measures, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale)
      loss /= n_imgs
      iou = Measures.calc_iou(measures, n_imgs, [0])
      print("finetune on", tag, idx, "/", start_step + n_finetune_steps, "loss:", loss, " iou:", iou, file=log.v5)


class InteractiveOneshotForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(InteractiveOneshotForwarder, self).__init__(engine)

  def forward(self, network, data, save_results=True, save_logits=False):
    self.network = network
    self.data = data
    tag = self.train_data.video_tag(0)
    print("finetuning on", tag, file=log.v4)

    #reset weights and optimizer for next video
    self.engine.try_load_weights()
    self.engine.reset_optimizer()
    self.data.set_video_idx(0)
    self._maybe_adjust_output_layer_for_multiple_objects()

    print("steps:", 0, file=log.v4)
    return self._oneshot_forward_video(0, save_logits)

  def _oneshot_forward_video(self, video_idx, save_logits):
    ys_armax = measures = None
    forward_interval = self.forward_interval
    if forward_interval > self.n_finetune_steps:
      forward_interval = self.n_finetune_steps
      n_partitions = 1
    else:
      n_partitions = int(ceil(float(self.n_finetune_steps) / forward_interval))
    if self.forward_initial:
      self._base_forward(self.engine.test_network, self.data, save_results=False, save_logits=False)
    for i in range(n_partitions):
      start = time.time()
      self._finetune(video_idx, n_finetune_steps=min(forward_interval, self.n_finetune_steps),
                     start_step=forward_interval * i)
      save_results = self.save_oneshot and i == n_partitions - 1
      save_logits_here = save_logits and i == n_partitions - 1
      ys_armax, measures = self._base_forward(self.network, self.data, save_results=save_results,
                           save_logits=save_logits_here)
      end = time.time()
      elapsed = end - start
      print("steps:", forward_interval * (i + 1), "elapsed", elapsed, file=log.v4)

    return ys_armax, measures

  def _base_forward(self, network, data, save_results, save_logits):
    interactive_image_forwarder = InteractiveImageForwarder.InteractiveImageForwarder(self.engine)
    ys_argmax, measures = interactive_image_forwarder.forward(network, data, save_results, save_logits)
    # super(OneshotForwarder, self).forward(network, data, save_results, save_logits)

    return ys_argmax, measures

  def _finetune(self, video_idx, n_finetune_steps, start_step=0):
    frame_ids = [0]
    if hasattr(self.train_data, "annotated_frame_ids"):
      frame_ids = self.train_data.annotated_frame_ids
    tag = self.train_data.video_tag(video_idx)
    self.train_data.set_video_idx(video_idx)
    for idx in range(start_step, start_step + n_finetune_steps):
      for frame_id in frame_ids:
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_id, with_annotations=True)
        loss_scale = 1.0
        loss, measures, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale)
        loss /= n_imgs
        iou = Measures.calc_iou(measures, n_imgs, [0])
        print("finetune on", tag, idx, "/", start_step + n_finetune_steps, "loss:", loss, " iou:", iou, file=log.v5)