import tensorflow as tf
import numpy

from ReID_net.datasets.COCO.COCO import COCODataset
import ReID_net.Constants as Constants

NUM_CLASSES = 90
N_MAX_DETECTIONS = 100


class COCODetectionDataset(COCODataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(COCODetectionDataset, self).__init__(config, subset, coord, fraction=fraction, num_classes=NUM_CLASSES)
    n_max_detections = max([len(x) for x in list(self.filename_to_anns.values())])
    assert n_max_detections <= N_MAX_DETECTIONS, n_max_detections

  def label_load_fn(self, img_path, label_path):
    def my_create_labels(im_path):
      return self.create_labels(im_path)
    bboxes, ids, classes = tf.py_func(my_create_labels, [img_path], [tf.float32, tf.int32, tf.int32])
    bboxes.set_shape((N_MAX_DETECTIONS, 4))
    ids.set_shape((N_MAX_DETECTIONS,))
    classes.set_shape((N_MAX_DETECTIONS,))
    labels = {Constants.BBOXES: bboxes, Constants.IDS: ids, Constants.CLASSES: classes}
    return labels

  def create_labels(self, img_path):
    anns = self.filename_to_anns[img_path.split("/")[-1]]
    #they need to be padded to N_MAX_DETECTIONS
    bboxes = numpy.zeros((N_MAX_DETECTIONS, 4), dtype="float32")
    # to avoid divison by 0:
    bboxes[:, [1, 3]] = 1
    ids = numpy.zeros(N_MAX_DETECTIONS, dtype="int32")
    classes = numpy.zeros(N_MAX_DETECTIONS, dtype="int32")

    if len(anns) > N_MAX_DETECTIONS:
      print("N_MAX_DETECTIONS not set high enough!")
      assert False, "N_MAX_DETECTIONS not set high enough!"

    for idx, ann in enumerate(anns):
      #TODO: ann["iscrowd"], ann["area"]
      x1 = ann["bbox"][0]
      y1 = ann["bbox"][1]
      box_width = ann["bbox"][2]
      box_height = ann["bbox"][3]
      x2 = x1 + box_width
      y2 = y1 + box_height
      bboxes[idx] = [y1, y2, x1, x2]
      ids[idx] = idx + 1
      # categories are from 1 to 90 in the annotations -> map to 0 to 89
      classes[idx] = ann["category_id"] - 1
    return bboxes, ids, classes
