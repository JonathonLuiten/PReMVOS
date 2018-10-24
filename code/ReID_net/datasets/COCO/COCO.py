import os

import numpy as np
import tensorflow as tf

from ReID_net.datasets.Dataset import ImageDataset
from ReID_net.datasets.Util import Reader
from ReID_net.Log import log

NUM_CLASSES = 2
COCO_DEFAULT_PATH = "/fastwork/" + os.environ['USER'] + "/mywork/data/coco/"
INPUT_SIZE = (None, None)
COCO_VOID_LABEL = 255
IGNORE_CLASSES = [0]


class COCODataset(ImageDataset):
  def __init__(self, config, subset, coord, fraction=1.0, ignore_classes=IGNORE_CLASSES, num_classes=NUM_CLASSES):
    super(COCODataset, self).__init__("coco", COCO_DEFAULT_PATH, num_classes, config, subset, coord, INPUT_SIZE,
                                      COCO_VOID_LABEL, fraction, label_load_fn=self.label_load_fn,
                                      img_load_fn=self.img_load_fn, ignore_classes=ignore_classes)
    if subset == "train":
      self.data_type = "train2014"
      self.filter_crowd_images = config.bool("filter_crowd_images", False)
      self.min_box_size = config.float("min_box_size", -1.0)
    else:
      self.data_type = "val2014"
      self.filter_crowd_images = False
      self.min_box_size = config.float("min_box_size_val", -1.0)
    # Use the minival split as done in https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md
    self.annotation_file = '%s/annotations/instances_%s.json' % (self.data_dir, subset)
    self.restricted_image_category_list = config.unicode_list("restricted_image_category_list", [])
    if len(self.restricted_image_category_list) == 0:
      self.restricted_image_category_list = None
    self.restricted_annotations_category_list = config.unicode_list("restricted_annotations_category_list", [])
    if len(self.restricted_annotations_category_list) == 0:
      self.restricted_annotations_category_list = None

    #either both of them or none should be specified for now to avoid unintuitive behaviour
    assert (self.restricted_image_category_list is None and self.restricted_annotations_category_list is None) or \
           (self.restricted_image_category_list is not None and self.restricted_annotations_category_list is not None),\
           (self.restricted_image_category_list, self.restricted_annotations_category_list)

    # only import this dependency on demand
    import pycocotools.coco as coco
    self.coco = coco.COCO(self.annotation_file)

    ann_ids = self.coco.getAnnIds([])
    self.anns = self.coco.loadAnns(ann_ids)
    self.label_map = {k - 1: v for k, v in list(self.coco.cats.items())}

    self.filename_to_anns = dict()
    self.build_filename_to_anns_dict()

  def build_filename_to_anns_dict(self):
    for ann in self.anns:
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = img[0]['file_name']
      if file_name in self.filename_to_anns:
        self.filename_to_anns[file_name].append(ann)
      else:
        self.filename_to_anns[file_name] = [ann]
        # self.filename_to_anns[file_name] = ann

    #exclude all images which contain a crowd
    if self.filter_crowd_images:
      self.filename_to_anns = {f: anns for f, anns in list(self.filename_to_anns.items())
                               if not any([an["iscrowd"] for an in anns])}
    #filter annotations with too small boxes
    if self.min_box_size != -1.0:
      self.filename_to_anns = {f: [ann for ann in anns if ann["bbox"][2] >= self.min_box_size and ann["bbox"][3]
                                   >= self.min_box_size] for f, anns in list(self.filename_to_anns.items())}

    #remove annotations with crowd regions
    self.filename_to_anns = {f: [ann for ann in anns if not ann["iscrowd"]]
                             for f, anns in list(self.filename_to_anns.items())}

    # restrict images to contain considered categories
    if self.restricted_image_category_list is not None:
      print("filtering images to contain categories", self.restricted_image_category_list, file=log.v1)
      self.filename_to_anns = {f: anns for f, anns in list(self.filename_to_anns.items())
                               if any([self.label_map[ann["category_id"] - 1]["name"]
                                       in self.restricted_image_category_list for ann in anns])}
      for cat in self.restricted_image_category_list:
        n_imgs_for_cat = sum([1 for anns in list(self.filename_to_anns.values()) if
                              any([self.label_map[ann["category_id"] - 1]["name"] == cat for ann in anns])])
        print("number of images containing", cat, ":", n_imgs_for_cat, file=log.v5)

    # restrict annotations to considered categories
    if self.restricted_annotations_category_list is not None:
      print("filtering annotations to categories", self.restricted_annotations_category_list, file=log.v1)
      self.filename_to_anns = {f: [ann for ann in anns if self.label_map[ann["category_id"] - 1]["name"]
                                   in self.restricted_annotations_category_list]
                               for f, anns in list(self.filename_to_anns.items())}

    #filter out images without annotations
    self.filename_to_anns = {f: anns for f, anns in list(self.filename_to_anns.items()) if len(anns) > 0}

    n_before = len(self.anns)
    self.anns = []
    for anns in list(self.filename_to_anns.values()):
      self.anns += anns
    n_after = len(self.anns)
    print("filtered annotations:", n_before, "->", n_after, file=log.v1)

  def get_filename_to_anns(self):
    return self.filename_to_anns

  def label_load_fn(self, img_path, label_path):
    def my_create_labels(im_path):
      return self.create_labels(im_path)
    label, old_label = tf.py_func(my_create_labels, [img_path], [tf.uint8, tf.uint8])
    labels = {"label": label, "old_label": old_label}
    return labels

  def img_load_fn(self, img_path):
    path = tf.string_split([img_path], '/').values[-1]
    #path = tf.Print(path, [path])
    img_dir = tf.cond(tf.equal(tf.string_split([path], '_').values[1], tf.constant("train2014")),
                      lambda: '%s/%s/' % (self.data_dir, "train2014"),
                      lambda: '%s/%s/' % (self.data_dir, "val2014"))
    path = img_dir + path
    return Reader.load_img_default(path)

  def create_labels(self, img_path):
    ann = self.filename_to_anns[img_path.split("/")[-1]]
    img = self.coco.loadImgs(ann[0]['image_id'])[0]

    height = img['height']
    width = img['width']

    label = np.zeros((height, width, 1))
    old_label = np.zeros((height, width, 1))

    label[:, :, 0] = self.coco.annToMask(ann[0])[:, :]
    if len(np.unique(label)) == 1:
      print("GT contains only background.")

    if 'bbox' in ann[0]:
      x = int(ann[0]['bbox'][0])
      y = int(ann[0]['bbox'][1])
      box_width = int(ann[0]['bbox'][2])
      box_height = int(ann[0]['bbox'][3])

      old_label[y:y+box_height, x:x+box_width, :] = 1

    return label.astype(np.uint8), old_label.astype(np.uint8)

  def read_inputfile_lists(self):
    img_dir = '%s/%s/' % (self.data_dir, self.data_type)
    #Filtering the image file names since some of them do not have annotations.
    imgs = [img_dir + fn for fn in list(self.filename_to_anns.keys())]
    img_ids = [anns[0]["image_id"] for fn, anns in list(self.filename_to_anns.items())]
    return [imgs, imgs, img_ids]

  def get_label_map(self):
    return self.label_map
