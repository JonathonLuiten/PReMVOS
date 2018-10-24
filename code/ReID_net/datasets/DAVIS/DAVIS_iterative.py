import scipy
import time

import numpy as np

import ReID_net.Measures as Measures
from ReID_net.Log import log
from ReID_net.datasets.DAVIS.DAVIS import read_image_and_annotation_list, group_into_sequences
from ReID_net.datasets.DAVIS.DAVIS_oneshot import DavisOneshotDataset, _load_flows
from ReID_net.datasets.Util.Reader import create_tensor_dict
from ReID_net.datasets.Util.Util import unique_list


def _load_frame(idx, im, an, imgs, subset, flow_dir, flow_into_past, flow_into_future, flow_as_angle, use_old_label):
  im_val = scipy.ndimage.imread(im) / 255.0

  flow_past, flow_future = _load_flows(idx, imgs, im_val.shape, flow_dir, flow_into_past, flow_into_future,
                                       flow_as_angle)

  if an != "":
    an_raw = scipy.ndimage.imread(an)
    if "adaptation" in an.split("/")[-1]:
      an_postproc = an_raw
      an_postproc[an_raw == 128] = 1
    else:
      an_postproc = an_raw / 255
      # if subset == "train":
        # Create a bounding box image
  else:
    an_postproc = np.zeros_like(im_val[:, :, 0])

  an_bbox = get_bounding_box(an_postproc, 1)
  an_bbox = np.expand_dims(an_bbox, 2)
  an_val = np.expand_dims(an_postproc, 2)
  tag_val = im

  if use_old_label:
    tensors = create_tensor_dict(unnormalized_img=im_val, label=an_val, tag=tag_val, flow_past=flow_past,
                                 flow_future=flow_future, old_label=an_bbox)
  else:
    tensors = create_tensor_dict(unnormalized_img=im_val, label=an_val, tag=tag_val, flow_past=flow_past,
                                 flow_future=flow_future)
  return tensors


def _load_video(imgs, ans, subset, flow_dir=None, flow_into_past=False, flow_into_future=False, flow_as_angle=False,
                use_old_label=True):
  video = []
  for idx_, (im_, an_) in enumerate(zip(imgs, ans)):
    tensors_ = _load_frame(idx_, im_, an_, imgs, subset, flow_dir, flow_into_past, flow_into_future, flow_as_angle,
                           use_old_label)
    video.append(tensors_)

  # from joblib import Parallel, delayed
  # video = Parallel(n_jobs=20, backend="threading")(
  #  delayed(_load_frame)(idx_, im_, an_, imgs, flow_dir, flow_into_past, flow_into_future, flow_as_angle)
  #  for idx_, (im_, an_) in enumerate(zip(imgs, ans)))

  return video


def get_bounding_box(mask, inst):
  mask = np.copy(mask)
  rows = np.where(mask == inst)[0]
  cols = np.where(mask == inst)[1]
  if len(rows) > 0 and len(cols) > 0:
    rmin = rows.min()
    rmax = rows.max()
    cmin = cols.min()
    cmax = cols.max()

    mask[rmin:rmax, cmin:cmax] = 1.0
  return mask


class DavisIterativeDataset(DavisOneshotDataset):
  def __init__(self, config, subset, use_old_label):
    super(DavisIterativeDataset, self).__init__(config, subset, use_old_label)
    self.use_old_label = use_old_label
    self.high_threshold=0.95
    self.low_threshold=0.1

  def load_videos(self, fn, data_dir, video_range=None):
    load_adaptation_data = self.adaptation_model != ""
    if load_adaptation_data:
      assert self.config.str("task") == "offline", self.config.str("task")
    elif DavisOneshotDataset._video_data is not None:
      # use this cache only if not doing offline adaptation!
      return DavisOneshotDataset._video_data

    print("loading davis dataset...", file=log.v4)
    imgs, ans = read_image_and_annotation_list(fn, data_dir)
    video_tags = unique_list([im.split("/")[-2] for im in imgs])
    imgs_seqs = group_into_sequences(imgs)
    ans_seqs = group_into_sequences(ans)

    if load_adaptation_data and self.subset == "train":
      # change annotations from groundtruth to adaptation data
      for video_tag, ans in zip(video_tags, ans_seqs):
        for idx in range(1, len(ans)):
          ans[idx] = "forwarded/" + self.adaptation_model + "/valid/" + video_tag + ("/adaptation_%05d.png" % idx)

    start = time.time()
    videos = [None] * len(imgs_seqs)
    if video_range is None:
      video_range = [0, len(imgs_seqs)]

    from joblib import Parallel, delayed
    videos[video_range[0]:video_range[1]] = Parallel(n_jobs=20, backend="threading")(delayed(_load_video)(
      imgs, ans, self.subset, self.flow_dir, self.flow_into_past, self.flow_into_future, self.flow_as_angle) for
                                                                                     (imgs, ans) in
                                                                                     zip(imgs_seqs, ans_seqs)[
                                                                                     video_range[0]:video_range[1]])

    # videos[video_range[0]:video_range[1]] = [_load_video(
    # imgs, ans, self.flow_dir, self.flow_into_past, self.flow_into_future, self.flow_as_angle) for
    #                               (imgs, ans) in zip(imgs_seqs, ans_seqs)[video_range[0]:video_range[1]]]

    DavisOneshotDataset._video_data = (video_tags, videos)
    end = time.time()
    elapsed = end - start
    print("loaded davis in", elapsed, "seconds", file=log.v4)
    return DavisOneshotDataset._video_data

  def feed_dict_for_video_frame(self, frame_idx, with_annotations, old_mask=None):
    """

    :param frame_idx: 
    :param with_annotations: 
    :param old_mask: The target lebel is set according to the logit provided, after a few postprocessing steps.
    :return: 
    """
    tensors = self._get_video_data()[frame_idx].copy()
    feed_dict = {self.img_placeholder: tensors["unnormalized_img"], self.tag_placeholder: tensors["tag"]}
    if with_annotations:
      if old_mask is not None:
        label = self.postprocess(old_mask[:, :, 1], frame_idx)
        feed_dict[self.label_placeholder] = np.expand_dims(label, 2)
      else:
        feed_dict[self.label_placeholder] = tensors["label"]

    if "old_label" in tensors:
      feed_dict[self.old_label_placeholder] = tensors['old_label']

    if "flow_past" in tensors:
      feed_dict[self.flow_into_past_placeholder] = tensors["flow_past"]
    if "flow_future" in tensors:
      feed_dict[self.flow_into_future_placeholder] = tensors["flow_future"]

    return feed_dict

  def change_label_to_bbox(self):
    """
        Change the target labels to bounding box
    """
    for frame_id in range(0, len(self._get_video_data())):
      annotation = get_bounding_box(self._get_video_data()[frame_id]['label'][:, :, 0], 1)
      self._get_video_data()[frame_id]['label'] = annotation[:, :, np.newaxis]

  def set_video_annotation(self, logits):
    """
    Use the annotations provided, as the label, after a few post processing steps.
    :param annotations: A list of annotations to be used as target labels
    """
    for frame_id in range(0, len(self._get_video_data())):
      annotation = self.postprocess(logits[frame_id][:, :, 1], frame_id)
      self._get_video_data()[frame_id]['label'] = annotation[:, :, np.newaxis]

  def postprocess(self, logit, frame_id):
    ann_postproc = np.zeros_like(logit)
    ann_postproc[np.where(logit > self.high_threshold)] = 1

    if 'old_label' in self._get_video_data()[frame_id]:
      bbox_initial = self._get_video_data()[frame_id]['old_label'][:, :, 0]
      bbox_ann = get_bounding_box(ann_postproc, 1)
      iou = Measures.compute_iou_for_binary_segmentation(bbox_ann, bbox_initial)
      # Postprocess 1: set annotation to bounding box of the object if IOU is less than 50%
      #if iou < 0.5:
      #  ann_postproc = bbox_ann

      # Postprocess 2: reset all pixels outside the bounding box
      ann_postproc = np.logical_and(bbox_initial,ann_postproc).astype(int)
      
      # set annotation within the threshold range to ignore pixels.
      ann_postproc[np.where(np.logical_and(logit > self.low_threshold, logit < self.high_threshold))] = self.void_label()

    return ann_postproc
