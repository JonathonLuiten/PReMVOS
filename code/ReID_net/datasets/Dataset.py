from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from skimage.draw import circle
import ReID_net.Constants as Constants
import ReID_net.datasets.Util.Util as Util
from ReID_net.Log import log
from ReID_net.datasets.Augmentors import parse_augmentors
from ReID_net.datasets.Util.Batch import create_batch_dict
from ReID_net.datasets.Util.Normalization import unnormalize
from ReID_net.datasets.Util.Reader import read_images_from_disk, load_label_default, load_img_default
from ReID_net.datasets.Util.Resize import parse_resize_mode, ResizeMode


class Dataset(object, metaclass=ABCMeta):
  def __init__(self, subset):
    self.summaries = []
    self.subset = subset

  def _get_resize_params(self, subset, input_size_default, resize_mode_default=ResizeMode.Unchanged):
    if subset == "valid":
      subset = "val"
    resize_mode_string = self.config.str("resize_mode_" + subset, "")
    # use same as for train if nothing specified
    if resize_mode_string == "":
      resize_mode_string = self.config.str("resize_mode_train", "")
    if resize_mode_string == "":
      resize_mode = resize_mode_default
    else:
      resize_mode = parse_resize_mode(resize_mode_string)
    input_size = self.config.int_list("input_size_" + subset, [])
    #use same as for train if nothing specified
    if len(input_size) == 0:
      if resize_mode == ResizeMode.Unchanged:
        input_size = [None, None]
      else:
        input_size = self.config.int_list("input_size_train", [])
    if len(input_size) == 0:
      input_size = input_size_default
    if resize_mode == ResizeMode.RandomResize:
      input_size = [None, None]
    return resize_mode, input_size

  def _parse_augmentors_and_shuffle(self):
    if self.subset == "train":
      shuffle = True
      augmentor_strs = self.config.unicode_list("augmentors_train", [])
      augmentors = parse_augmentors(augmentor_strs, self.void_label())
      if len(augmentors) == 0:
        print("warning, no data augmentors used on train", file=log.v1)
    else:
      shuffle = False
      augmentor_strs = self.config.unicode_list("augmentors_val", [])
      augmentors = parse_augmentors(augmentor_strs, self.void_label())
    return augmentors, shuffle

  @abstractmethod
  def num_classes(self):
    pass

  @abstractmethod
  def num_examples_per_epoch(self):
    pass

  # should contain inputs, labels, tags, maybe raw_labels, index_img
  @abstractmethod
  def create_input_tensors_dict(self, batch_size):
    pass

  @abstractmethod
  def void_label(self):
    pass


class ImageDataset(Dataset):
  def __init__(self, dataset_name, default_path, num_classes, config, subset, coord, image_size, void_label=255,
               fraction=1.0, label_postproc_fn=lambda x: x, label_load_fn=load_label_default,
               img_load_fn=load_img_default, ignore_classes=[]):
    super(ImageDataset, self).__init__(subset)
    self._num_classes = num_classes
    self._void_label = void_label
    assert subset in ("train", "valid"), subset
    if subset == "train":
      self.epoch_length = config.int("epoch_length", - 1)
    else:
      self.epoch_length = -1
    self.config = config
    self.data_dir = config.str(dataset_name + "_data_dir", default_path)
    self.coord = coord
    self.image_size = image_size
    self.inputfile_lists = None
    self.fraction = fraction
    self.label_postproc_fn = label_postproc_fn
    self.label_load_fn = label_load_fn
    self.img_load_fn = img_load_fn
    self.use_summaries = self.config.bool("use_summaries", False)
    self.ignore_classes = ignore_classes

  def _load_inputfile_lists(self):
    if self.inputfile_lists is not None:
      return
    self.inputfile_lists = self.read_inputfile_lists()
    # make sure all lists have the same length
    assert all([len(l) == len(self.inputfile_lists[0]) for l in self.inputfile_lists])
    if self.fraction < 1.0:
      n = int(self.fraction * len(self.inputfile_lists[0]))
      self.inputfile_lists = [l[:n] for l in self.inputfile_lists]

  def create_input_tensors_dict(self, batch_size):
    self._load_inputfile_lists()
    resize_mode, input_size = self._get_resize_params(self.subset, self.image_size, ResizeMode.Unchanged)
    augmentors, shuffle = self._parse_augmentors_and_shuffle()

    #inputfile_tensors = [tf.convert_to_tensor(l, dtype=tf.string) for l in self.inputfile_lists]
    inputfile_tensors = [tf.convert_to_tensor(l) for l in self.inputfile_lists]
    queue = tf.train.slice_input_producer(inputfile_tensors, shuffle=shuffle)

    tensors_dict, summaries = self._read_inputfiles(queue, resize_mode, input_size, augmentors)
    tensors_dict, summ = create_batch_dict(batch_size, tensors_dict)
    if summ is not None:
      summaries.append(summ)
    if Constants.BBOXES in tensors_dict and Constants.IDS in tensors_dict and Constants.CLASSES in tensors_dict:
      assert "labels" not in tensors_dict
      tensors_dict["labels"] = (tensors_dict[Constants.BBOXES], tensors_dict[Constants.IDS],
                                tensors_dict[Constants.CLASSES])
      if Constants.IGNORE_REGIONS in tensors_dict:
        tensors_dict["labels"] += (tensors_dict[Constants.IGNORE_REGIONS],)
        if Constants.SCENE_INFOS in tensors_dict:
          tensors_dict["labels"] += (tensors_dict[Constants.SCENE_INFOS],)

    if self.use_summaries:
      summaries += self._create_summaries(tensors_dict)

    self.summaries += summaries
    return tensors_dict

  def _create_summaries(self, tensors_dict):
    inputs = tensors_dict["inputs"]
    input_imgs = unnormalize(inputs[:, :, :, :3])
    # count is incremented after each summary creation. This helps us to keep track of the number of channels
    start = 3
    summaries = []
    # Old label would be present either as a single channel, or along with 2 other distance transform channels.
    if inputs.get_shape()[-1] in [4,6,8]:
      if Constants.OLD_LABEL_AS_DT in tensors_dict:
        [input_imgs] = tf.py_func(self.add_clicks, [tf.concat([input_imgs, inputs[:, :, :, start:start + 1]],
                                                              axis=3), 'g'], [tf.float32])
      else:
        [old_label, input_imgs] = tf.py_func(self.get_masked_image, [input_imgs, inputs[:, :, :, start:start + 1]],
                                            [tf.float32, tf.float32])
        
      summ = tf.summary.image("old_labels", inputs[:, :, :, start:start + 1])
      summaries.append(summ)
      start += 1

    # Add clicks to the image so that they can be viewed in tensorboard
    if inputs.get_shape()[-1] > 6:
      [input_imgs] = tf.py_func(self.add_clicks, [tf.concat([input_imgs, inputs[:, :, :, start:start + 1]],
                                                  axis=3), 'r'], [tf.float32])
      [input_imgs] = tf.py_func(self.add_clicks, [tf.concat([input_imgs, inputs[:, :, :, start + 1:start + 2]],
                                                  axis=3), 'r'], [tf.float32])
      [input_imgs] = tf.py_func(self.add_clicks, [tf.concat([input_imgs, inputs[:, :, :, start + 2:start + 3]],
                                                  axis=3), 'g'], [tf.float32])
      [input_imgs] = tf.py_func(self.add_clicks, [tf.concat([input_imgs, inputs[:, :, :, start + 3:start + 4]],
                                                  axis=3), 'g'], [tf.float32])
    elif inputs.get_shape()[-1] > 4:
      [input_imgs] = tf.py_func(self.add_clicks, [tf.concat([input_imgs, inputs[:, :, :, start:start + 1]],
                                                  axis=3), 'r'], [tf.float32])
      [input_imgs] = tf.py_func(self.add_clicks, [tf.concat([input_imgs, inputs[:, :, :, start + 1:start + 2]],
                                                  axis=3), 'g'], [tf.float32])

    # bounding boxes
    if Constants.BBOXES in tensors_dict:
      bboxes = tensors_dict[Constants.BBOXES]
      # permute y1, y2, x1, x2 -> y1, x1, y2, x1
      bboxes = tf.stack([bboxes[..., 0], bboxes[..., 2], bboxes[..., 1], bboxes[..., 3]], axis=-1)
      # normalize bboxes to [0..1]
      height = tf.shape(input_imgs)[1]
      width = tf.shape(input_imgs)[2]
      bboxes = tf.cast(bboxes, tf.float32) / tf.cast(tf.stack([height, width, height, width], axis=0), tf.float32)
      imgs_with_bboxes = tf.image.draw_bounding_boxes(input_imgs, bboxes)
      summ = tf.summary.image("inputs", imgs_with_bboxes)
      summaries.append(summ)
    else:
      summ = tf.summary.image("inputs", input_imgs)
      summaries.append(summ)

    if not isinstance(tensors_dict["labels"], tuple):
      summ = tf.summary.image("labels", tensors_dict["labels"] * 255)  # will only work well for binary segmentation
      summaries.append(summ)

    # Append the distance transforms, if they are available.
    if inputs.get_shape()[-1] > 6:
      # Get negative distance transform from the extra input channels.
      summ = tf.summary.image(Constants.DT_NEG, inputs[:, :, :, start:start + 1])
      summaries.append(summ)
      summ = tf.summary.image(Constants.DT_NEG, inputs[:, :, :, start+1:start + 2])
      summaries.append(summ)
      summ = tf.summary.image(Constants.DT_POS, inputs[:, :, :, start + 2:start + 3])
      summaries.append(summ)
      summ = tf.summary.image(Constants.DT_POS, inputs[:, :, :, start + 3:start + 4])
      summaries.append(summ)
    elif inputs.get_shape()[-1] > 4:
      # Get negative distance transform from the extra input channels.
      summ = tf.summary.image(Constants.DT_NEG, inputs[:, :, :, start:start + 1])
      summaries.append(summ)
      summ = tf.summary.image(Constants.DT_POS, inputs[:, :, :, start + 1:start + 2])
      summaries.append(summ)

    return summaries

  # default implementation, should in many cases be overwritten
  def _read_inputfiles(self, queue, resize_mode, input_size, augmentors):
    tensors, summaries = read_images_from_disk(queue, input_size, resize_mode, label_postproc_fn=self.label_postproc_fn,
                                               augmentors=augmentors, label_load_fn=self.label_load_fn,
                                               img_load_fn=self.img_load_fn)
    return tensors, summaries

  def add_clicks(self, inputs, c):
    out_imgs = None

    for input in inputs:
      img = input[:, :, :3]
      # Radius of the point to be diplayed
      r=3
      pts = np.where(input[:,:,3] == 0)
      pts_zipped = list(zip(pts[0], pts[1]))
      if len(pts[0]) > 0:
        for pt in pts_zipped:
          if r < pt[0] < img.shape[0] - r and r < pt[1] < img.shape[1] - r:
            rr, cc = circle(pt[0], pt[1], 5, img.shape)
            img[rr, cc, :] = [np.max(img), np.min(img), np.min(img)] if c == 'r' \
                else [np.min(img), np.min(img), np.max(img)]

      img = img[np.newaxis, :, :, :]
      if out_imgs is None:
        out_imgs = img
      else:
        out_imgs = np.concatenate((out_imgs, img), axis = 0)

    return out_imgs.astype(np.float32)

  def get_masked_image(self, inputs, old_labels):
    out_imgs = None
    out_labels = None
    for input_img, old_label_unmodified in zip(inputs, old_labels):
      if len(np.unique(old_label_unmodified)) > 2:
        old_label = np.where(old_label_unmodified*255 <= 128, 1, 0)
      else:
        old_label = old_label_unmodified

      out_img = Util.get_masked_image(input_img, old_label[:, :, 0])
      out_img = out_img[np.newaxis, :, :, :]
      out_label = old_label[np.newaxis, :, :, :]
      if out_imgs is None:
        out_imgs = out_img
      else:
        out_imgs = np.concatenate((out_imgs, out_img), axis = 0)

      if out_labels is None:
        out_labels = out_label
      else:
        out_labels = np.concatenate((out_labels, out_label), axis = 0)

    return out_labels.astype(np.float32), out_imgs.astype(np.float32)

  @abstractmethod
  def read_inputfile_lists(self):
    pass

  def num_examples_per_epoch(self):
    if self.epoch_length != -1:
      return self.epoch_length
    else:
      self._load_inputfile_lists()
      return len(self.inputfile_lists[0])

  def num_classes(self):
    return self._num_classes

  def void_label(self):
    return self._void_label

  def ignore_classes(self):
    return self.ignore_classes
