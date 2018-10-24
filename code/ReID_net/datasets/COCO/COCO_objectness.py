from ReID_net.datasets.COCO.COCO import COCODataset
import os
import tensorflow as tf
import numpy as np

NUM_CLASSES = 2
COCO_DEFAULT_PATH = "/fastwork/" + os.environ['USER'] + "/mywork/data/coco/"
INPUT_SIZE = (None, None)
COCO_VOID_LABEL = 255


class CocoObjectnessDataset(COCODataset):
  def label_load_fn(self, img_path, label_path):
    def my_create_labels(im_path):
      return self.create_labels(im_path)
    label, old_label = tf.py_func(my_create_labels, [img_path], [tf.uint8, tf.uint8])
    labels = {"label": label}
    return labels

  def create_labels(self, img_path):
    anns = self.filename_to_anns[img_path.split("/")[-1]]
    img = self.coco.loadImgs(anns[0]['image_id'])[0]
    height = img['height']
    width = img['width']

    label = np.zeros((height, width, 1))
    old_label = np.zeros((height, width, 1))

    for ann in anns:
      mask = self.coco.annToMask(ann)[:, :]
      label[mask == 1] = 1

    return label.astype(np.uint8), old_label.astype(np.uint8)
