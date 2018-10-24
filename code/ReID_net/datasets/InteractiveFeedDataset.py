import glob
from abc import abstractmethod

import numpy as np

import ReID_net.Constants as Constants
from ReID_net.datasets.DAVIS.DAVIS_iterative import _load_video
from ReID_net.datasets.FeedDataset import FeedImageDataset, OneshotImageDataset
from ReID_net.datasets.PascalVOC.PascalVOC_interactive import PascalVOCInteractiveDataset
from ReID_net.datasets.PascalVOC.PascalVOC_interactive_correction import create_old_mask
from ReID_net.datasets.Util.Reader import create_tensor_dict
from ReID_net.datasets.Util.Util import username
import tensorflow as tf

NUM_CLASSES = 2
VOID_LABEL = 255
INPUT_SIZE = (None, None)
DAVIS_IMAGE_SIZE = (480, 854)


class InteractiveFeedDataset(FeedImageDataset):

  def __init__(self, config, subset, coord, data_dir, fraction=1.0, num_classes = NUM_CLASSES,
              void_label = VOID_LABEL, INPUT_SIZE = INPUT_SIZE, use_old_label = False, use_clicks = True):
    super(InteractiveFeedDataset, self).__init__(config, num_classes, void_label, subset, INPUT_SIZE,
                                           use_old_label = use_old_label, use_clicks = use_clicks)
    self.coord = coord
    self.data_dir = data_dir

  def num_examples_per_epoch(self):
    return 1

  def get_feed_dict(self):
    return self.feed_dict

  @abstractmethod
  def read_inputfile_lists(self):
    pass


class PascalVOCInteractiveFeed(InteractiveFeedDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    data_dir = "/fastwork/" + username() + "/mywork/data/PascalVOC/benchmark_RELEASE/dataset/"
    self.use_old_label = config.bool("use_old_label", False)
    super(PascalVOCInteractiveFeed, self).__init__(config, subset, coord, data_dir, fraction,
                                                   NUM_CLASSES, VOID_LABEL, INPUT_SIZE,
                                                   use_old_label=self.use_old_label, use_clicks=True)
    self.fraction = fraction

  def read_inputfile_lists(self):
    pascalVOCInteractive =  PascalVOCInteractiveDataset(self.config,
                                                        self.subset,
                                                        self.coord,
                                                        fraction=self.fraction)

    input_file_list = pascalVOCInteractive.read_inputfile_lists()

    if self.fraction < 1.0:
      n = int(self.fraction * len(input_file_list[0]))
      input_file_list = [l[:n] for l in input_file_list]

    return input_file_list


class PascalVOCInteractiveCorrectionFeed(InteractiveFeedDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    data_dir = "/fastwork/" + username() + "/mywork/data/PascalVOC/benchmark_RELEASE/dataset/"
    super(PascalVOCInteractiveCorrectionFeed, self).__init__(config, subset, coord, data_dir, fraction,
                                                             NUM_CLASSES, VOID_LABEL, INPUT_SIZE,
                                                             use_old_label=True, use_clicks=True)
    self.fraction = fraction
    self.old_label_as_distance_transform = config.bool("old_label_as_distance_transform", True)

  def create_feed_dict(self, img, label, tag, old_label=None, flow_past=None, flow_future=None,
                       u0=None, u1=None):
    if old_label is not None and self.old_label_as_distance_transform:
      old_label = create_old_mask(old_label[:, :, 0])
      old_label = old_label[:, :, np.newaxis]

    tensors = create_tensor_dict(unnormalized_img=img, label=label, tag=tag, old_label=old_label,
                                 flow_past=flow_past, flow_future=flow_future,
                                 u0=u0, u1=u1)

    self.feed_dict = {self.img_placeholder: tensors["unnormalized_img"],
                      self.label_placeholder: tensors["label"],
                      self.tag_placeholder: tensors["tag"]}

    if "old_label" in tensors:
      self.feed_dict[self.old_label_placeholder] = tensors["old_label"]
    if "flow_past" in tensors:
      self.feed_dict[self.flow_into_past_placeholder] = tensors["flow_past"]
    if "flow_future" in tensors:
      self.feed_dict[self.flow_into_future_placeholder] = tensors["flow_future"]
    if Constants.DT_NEG in tensors:
      self.feed_dict[self.u0_placeholder] = tensors[Constants.DT_NEG]
    if Constants.DT_POS in tensors:
      self.feed_dict[self.u1_placeholder] = tensors[Constants.DT_POS]

    return self.feed_dict

  def read_inputfile_lists(self):
    pascalVOCInteractive =  PascalVOCInteractiveDataset(self.config,
                                                        self.subset,
                                                        self.coord,
                                                        fraction=self.fraction)

    input_file_list = pascalVOCInteractive.read_inputfile_lists()

    if self.fraction < 1.0:
      n = int(self.fraction * len(input_file_list[0]))
      input_file_list = [l[:n] for l in input_file_list]

    return input_file_list


class InteractiveOneShot(OneshotImageDataset):
  _cache = None

  def __init__(self, config, subset, video_data_dir=None, num_classes = NUM_CLASSES, void_label=VOID_LABEL,
               image_size=(None, None)):
    self.use_old_label = config.bool("use_old_label", False)
    super(InteractiveOneShot, self).__init__(config, num_classes, void_label, subset=subset,
                                             image_size=image_size, use_old_label=self.use_old_label)
    self.data_dir = video_data_dir
    self.videos = None
    self.annotated_frame_ids = []
    # self._video_tags, self.videos = self.load_data()

  def load_data(self):
    ims = glob.glob(self.data_dir[0] + "/*")
    ims = sorted(ims)

    if self.data_dir[1] is None:
      ans = np.repeat("", len(ims)).tolist()
    else:
      ans = sorted(glob.glob(self.data_dir[1] + "/*"))

    video = _load_video(imgs=ims, ans=ans, subset=self.subset, use_old_label=self.use_old_label)
    return self.data_dir, video

  def _get_video_data(self):
    if self.videos is None and InteractiveOneShot._cache is not None:
      self._video_tags, self.videos = InteractiveOneShot._cache[1]

    return self.videos

  def set_video_data_dir(self, video_data_dir, anns_data_dir = None):
    self.data_dir = (video_data_dir, anns_data_dir)
    if InteractiveOneShot._cache is not None and InteractiveOneShot._cache[0] == video_data_dir:
      self._video_tags, self.videos = InteractiveOneShot._cache[1]
    else:
      self._video_tags, self.videos = self.load_data()
      InteractiveOneShot._cache = (video_data_dir, (self._video_tags, self.videos))

  def set_video_annotation(self, frame_ids_annotation_dict):
    """

    :param frame_ids_annotation_dict: A dictionary that has frame ids as the keys and the corresponding annotation
                                      masks as their values
    """
    self.annotated_frame_ids = list(frame_ids_annotation_dict.keys())
    for frame_id in self.annotated_frame_ids:
      self._get_video_data()[frame_id]['label'] = frame_ids_annotation_dict[frame_id]


class PascalVOCRecursiveCorrectionFeed(PascalVOCInteractiveCorrectionFeed):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(PascalVOCRecursiveCorrectionFeed, self).__init__(config, subset, coord, fraction=fraction)
    self.u0_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))
    self.u1_placeholder = tf.placeholder(tf.float32, shape=(None, None, 2))
