import random

import numpy as np
import tensorflow as tf

import ReID_net.Constants as Constants
from ReID_net.datasets.COCO.COCO_instance import COCOInstanceDataset
from ReID_net.datasets.Util import Reader


class COCOInteractiveDataset(COCOInstanceDataset):

  def __init__(self, config, subset, coord, fraction=1.0):
    super(COCOInteractiveDataset, self).__init__(config, subset, coord, fraction=fraction)
    self.d_margin = 5
    self.strategies = [1, 2, 3]
    self.n_pairs = config.int("n_pairs", 3)
    self.dt_method = config.str(Constants.INTERACTIVE_DT_METHOD, "edt")

  def label_load_fn(self, img_path, label_path):
    label, old_label, raw_label, strategy = tf.py_func(self.load_label,
                                                       [img_path, label_path],
                                                       [tf.uint8, tf.uint8,
                                                        tf.uint8, tf.int64])

    labels = {"label": label, "raw_label": raw_label, Constants.STRATEGY: strategy,
              Constants.IGNORE_CLASSES: self.ignore_classes, Constants.USE_CLICKS:True}
    return labels

  def load_label(self, img_path, label_path):
    # Sample from the strategies here instead of the one provided in label path. This would 
    # ensure randomness of the same in case we are using just a single input pair. 
    strategy = random.sample(self.strategies, 1)[0]
    ann = self.filename_to_anns[img_path][0]
    ann_id = ann['id']
    all_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=ann['image_id']))
    img = self.coco.loadImgs(ann['image_id'])[0]

    height = img['height']
    width = img['width']

    label = np.zeros((height, width, 1))
    old_label = np.zeros((height, width, 1))

    label[:, :, 0] = self.coco.annToMask(ann)
    if len(np.unique(label)) == 1:
      print("GT contains only background.")

    # Set all other object instances
    label_unmodified = np.copy(label)

    # Load the negative object instances only when we have to sample clicks on negative objects.
    if strategy == 2:
      inst = 2
      for an in all_anns:
        # Ignore current object instance
        if an['id'] != ann_id:
          new_inst = self.coco.annToMask(an)
          label_unmodified[np.where(new_inst == 1)] = inst
          inst += 1

    old_label = old_label[:, :, np.newaxis]

    return label.astype(np.uint8), old_label.astype(np.uint8), label_unmodified.astype(np.uint8), strategy

  def img_load_fn(self, img_path):
    path = tf.string_split([img_path], ':').values[0]
    img_dir = '%s/%s/' % (self.data_dir, self.data_type)
    path = img_dir + path
    return Reader.load_img_default(path)

  def read_inputfile_lists(self):
    strategies_str = [str(strategy) for strategy in self.strategies]

    if self.subset == "train":
      imgs = np.repeat(list(self.filename_to_anns.keys()), self.n_pairs)
    else:
      imgs = list(self.filename_to_anns.keys())

    strategies_sampled = np.random.choice(strategies_str, len(imgs),
                                          p=np.repeat(1.0 / len(strategies_str), len(strategies_str)))
    strategies_sampled = list(strategies_sampled)

    return [imgs, strategies_sampled]
