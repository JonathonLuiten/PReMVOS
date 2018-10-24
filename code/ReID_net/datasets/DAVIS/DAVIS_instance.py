from ReID_net.datasets.DAVIS.DAVIS_oneshot import DavisOneshotDataset
from ReID_net.datasets.DAVIS.DAVIS import VOID_LABEL
import numpy as np


def get_bounding_box(mask, inst):
  mask = np.copy(mask)
  rows = np.where(mask == inst)[0]
  cols = np.where(mask == inst)[1]
  rmin = rows.min()
  rmax = rows.max()
  cmin = cols.min()
  cmax = cols.max()

  mask[rmin:rmax, cmin:cmax] = 1.0
  return mask


class DAVISInstanceDataset(DavisOneshotDataset):
  def __init__(self, config, subset, use_old_label):
    self.use_old_label = use_old_label
    super(DAVISInstanceDataset, self).__init__(config, subset, use_old_label)

  def feed_dict_for_video_frame(self, frame_idx, with_annotations, old_mask=None, train_on_background=False):
    tensors = self._get_video_data()[frame_idx].copy()
    feed_dict = {self.img_placeholder: tensors["unnormalized_img"], self.tag_placeholder: tensors["tag"]}
    if with_annotations:
      if train_on_background:
        bbox = get_bounding_box(tensors["label"], 1)
        feed_dict[self.label_placeholder] = np.where(bbox == 1, VOID_LABEL, bbox)
      else:
        feed_dict[self.label_placeholder] = tensors["label"]

    assert "old_mask" not in tensors
    if old_mask is not None:
      feed_dict[self.old_label_placeholder] = old_mask

    if "flow_past" in tensors:
      feed_dict[self.flow_into_past_placeholder] = tensors["flow_past"]
    if "flow_future" in tensors:
      feed_dict[self.flow_into_future_placeholder] = tensors["flow_future"]

    if self.use_old_label:
      feed_dict[self.old_label_placeholder] = get_bounding_box(tensors["label"], 1)

    return feed_dict
