import tensorflow as tf

from ReID_net.datasets.Dataset import Dataset
from ReID_net.datasets.Util.Reader import read_images_from_disk
from ReID_net.datasets.Util.Batch import create_batch_dict
from ReID_net.datasets.Util.Resize import ResizeMode
import ReID_net.Constants as Constants


def _create_labels(img_path, bboxes):
  # this dataset will not be used for training, so we can set ids and classes automatically
  ids = tf.range(tf.shape(bboxes)[0]) + 1
  classes = tf.zeros(tf.shape(ids), dtype=tf.int32)
  labels = {Constants.BBOXES: bboxes, Constants.IDS: ids, Constants.CLASSES: classes}
  return labels


class DetectionFeedDataset(Dataset):
  def __init__(self, config, subset, coord):
    super(DetectionFeedDataset, self).__init__(subset)
    self.config = config
    self.img_filename_placeholder = tf.placeholder(tf.string, shape=(), name="img_filename_placeholder")
    self.bboxes_placeholder = tf.placeholder(tf.float32, shape=(None, 4), name="label_placeholder")

  def create_input_tensors_dict(self, batch_size):
    assert batch_size == 1
    #TODO: get this from ReID_net.Config
    resize_mode, input_size = self._get_resize_params(self.subset, [None, None], ResizeMode.DetectionFixedSizeForEval)
    queue = [self.img_filename_placeholder, self.bboxes_placeholder]
    tensors_dict, summaries = read_images_from_disk(queue, input_size, resize_mode, label_load_fn=_create_labels)
    tensors_dict, summ = create_batch_dict(batch_size, tensors_dict)
    tensors_dict["labels"] = (tensors_dict[Constants.BBOXES], tensors_dict[Constants.IDS],
                              tensors_dict[Constants.CLASSES])
    return tensors_dict

  def num_classes(self):
    # keep it consitent with coco for the moment
    return 90

  def num_examples_per_epoch(self):
    return 1

  def void_label(self):
    return 255
