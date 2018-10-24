import ReID_net.Measures as Measures
import ReID_net.Util as Util
from ReID_net.Log import log
import numpy
from ReID_net.Forwarding.OneshotForwarder import OneshotForwarder


#for legacy adaptation images handling (can be removed at some point)
def create_bad_labels():
  from scipy.ndimage.morphology import distance_transform_edt
  thresholds = [100, 150, 175, 180, 190, 200, 220, 250]
  l = []
  for threshold in thresholds:
    x = numpy.ones((480, 854))
    dt = distance_transform_edt(x)
    z = numpy.zeros((480, 854))
    z[:] = 255
    negatives = dt > threshold
    z[negatives] = 0
    l.append(numpy.expand_dims(z, axis=2))
  return l

bad_labels = create_bad_labels()


class OfflineAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(OfflineAdaptingForwarder, self).__init__(engine)
    self.offline_adaptation_interval = self.config.int("offline_adaptation_interval")
    self.adaptation_loss_scale = self.config.float("adaptation_loss_scale", 0.1)

  def _finetune(self, video_idx, n_finetune_steps, start_step=0):
    print("offline finetuning...")
    tag = self.train_data.video_tag(video_idx)
    self.train_data.set_video_idx(video_idx)
    n_frames = self.train_data.num_examples_per_epoch()

    #sampling without replacement
    to_sample = list(range(1, n_frames))

    for step_idx in range(start_step, start_step + n_finetune_steps):
      if step_idx % self.offline_adaptation_interval == 0:
        found = False
        frame_idx = feed_dict = None
        while not found:
          if len(to_sample) == 0:
            to_sample = list(range(1, n_frames))
          frame_idx = numpy.random.choice(to_sample)
          feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
          label = feed_dict[self.train_data.get_label_placeholder()]
          
          is_bad_label = (label == 255).all()
          #legacy adaptation images handling
          for bad_label in bad_labels:
            if (label.shape == bad_label.shape) and (label == bad_label).all():
              is_bad_label = True
          
          if is_bad_label:
            print("sequence", tag, "frame", frame_idx, "has bad label, selecting new one...")
          else:
            found = True
          to_sample.remove(frame_idx)
        loss_scale = self.adaptation_loss_scale
        print("using adaptation sample: frame", frame_idx)
      else:
        feed_dict = self.train_data.feed_dict_for_video_frame(0, with_annotations=True)
        loss_scale = 1.0
      loss, measures, n_imgs = self.trainer.train_step(epoch=step_idx, feed_dict=feed_dict, loss_scale=loss_scale)
      loss /= n_imgs
      iou = Measures.calc_iou(measures, n_imgs, [0])
      print("finetune on", tag, step_idx, "/", start_step + n_finetune_steps, "loss:", loss, " err:", iou)
