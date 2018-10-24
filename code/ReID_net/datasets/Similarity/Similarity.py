import tensorflow as tf
import numpy
from math import ceil

from ReID_net.datasets.Dataset import Dataset
from ReID_net.datasets.Util.Reader import load_image_tensorflow
from ReID_net.datasets.Util.Resize import resize_image
from ReID_net.datasets.Util.Util import smart_shape, username
from ReID_net.datasets.Augmentors import apply_augmentors
from ReID_net.datasets.Util.Normalization import normalize, unnormalize

SIMILARITY_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/CUHK03/"
DEFAULT_INPUT_SIZE = [128, 128]
SIMILARITY_VOID_LABEL = 255


class SimilarityDataset(Dataset):
  def __init__(self, config, subset, coord, annotations, n_train_ids, jpg=True):
    super(SimilarityDataset, self).__init__(subset)
    assert subset in ("train", "valid"), subset
    self.jpg = jpg
    self.config = config
    self.subset = subset
    self.coord = coord
    self.model = config.str("model", "")
    self.annotations = annotations

    self.input_size = config.int_list("input_size", DEFAULT_INPUT_SIZE)
    self.input_size = tuple(self.input_size)
    self.batching_mode = config.str("batching_mode", "pair")
    assert self.batching_mode in ("single", "pair", "group", "eval"), self.batching_mode
    self.validation_mode = config.str("validation_mode", "embedding")
    assert self.validation_mode in ("embedding", "similarity"), self.validation_mode
    self.group_size = config.int("group_size", 2)
    self.pair_ratio = config.int("pair_ratio", 1)
    self.augmentors, _ = self._parse_augmentors_and_shuffle()
    self.context_region_factor = config.float("context_region_factor", 1.2)
    if self.subset != "train":
      context_region_factor_val = config.float("context_region_factor_val", -1.0)
      if context_region_factor_val != -1.0:
        self.context_region_factor = context_region_factor_val
    self.use_summaries = self.config.bool("use_summaries", False)

    self.epoch_length = config.int("epoch_length", 1000)
    if subset != "train":
      epoch_length = config.int("epoch_length_val", -1)
      if epoch_length != -1:
        self.epoch_length = epoch_length

    if self.batching_mode == "eval":
      assert len(self.augmentors) == 0, len(self.augmentors)
      self.epoch_length = len(annotations)

    self.n_classes = config.int("num_classes", None)
    self.num_train_id = n_train_ids

    self.file_names_list = [ann["img_file"] for ann in annotations]
    has_tags = "tag" in annotations[0]
    if has_tags:
      self.tags_list = [ann["tag"] for ann in annotations]
    else:
      self.tags_list = self.file_names_list
    self.bboxes_list = numpy.array([ann["bbox"] for ann in annotations], dtype="float32")
    cat_ids = [ann["category_id"] for ann in annotations]
    self.class_labels_list = numpy.array(cat_ids, dtype="int32")

    train_id_list, train_counts = numpy.unique(sorted(cat_ids), return_counts=True)
    # print(set(numpy.arange(0,5952))-set(train_id_list))
    # print(len(train_id_list),len(train_counts))
    # print("started seeming long thing")
    # import time
    # t = time.time()
    self.indices_for_classes = [[idx for idx, id_ in enumerate(cat_ids) if id_ == cat_id] for cat_id in train_id_list]
    # print("finished seeming long thing",time.time()-t)
    self.train_counts = tf.constant(train_counts.astype(numpy.int32))

    self.idx_placeholder = tf.placeholder(tf.int32, (4,), "idx")
    self.test_case = tf.placeholder(tf.string)
    self.use_end_network = tf.placeholder(tf.bool)

  def num_classes(self):
    return self.n_classes

  def _create_inputs_for_eval(self, batch_size):
    #for now require batch size of 1, which will allow us to output crops of different sizes for visualization
    # assert batch_size == 1

    tf_fns = tf.constant(self.file_names_list, tf.string)
    tf_tags = tf.constant(self.tags_list, tf.string)
    tf_bboxes = tf.constant(self.bboxes_list, tf.float32)
    tf_class_labels_list = tf.constant(self.class_labels_list, tf.int32)

    def load_fn(fn_, tag, bbox_, class_label_):
      if batch_size == 1:
        img, _, img_raw = self._load_crop_helper(fn_, bbox_)
      else:
        img, img_raw, _ = self._load_crop_helper(fn_, bbox_)
      return img, img_raw, tag, class_label_

    USE_DATASET_API = True
    if USE_DATASET_API:
      dataset = tf.data.Dataset.from_tensor_slices((tf_fns, tf_tags, tf_bboxes, tf_class_labels_list))
      # dataset = dataset.map(load_fn)
      dataset = dataset.map(load_fn,num_parallel_calls=32)
      #dataset = dataset.batch(batch_size)
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()
      imgs_norm, imgs_raw, tags, class_labels = next_element
    else:
      #alternative with slice_input_producer
      fn, tag_, bbox, class_label = tf.train.slice_input_producer((tf_fns, tf_tags, tf_bboxes, tf_class_labels_list),
                                                            num_epochs=1, shuffle=False)
      imgs_norm, imgs_raw, tags, class_labels = load_fn(fn, tag_, bbox, class_label)

    #expand for batch size of 1
    if batch_size == 1:
      imgs_norm = tf.expand_dims(imgs_norm, axis=0)
      imgs_raw = tf.expand_dims(imgs_raw, axis=0)
      tags = tf.expand_dims(tags, axis=0)
      class_labels = tf.expand_dims(class_labels, axis=0)
    else:
      imgs_norm, imgs_raw, tags, class_labels = tf.train.batch([imgs_norm, imgs_raw, tags, class_labels], batch_size,
                                                               num_threads=32, capacity=10 * batch_size,
                                                               allow_smaller_final_batch=True)
      print(imgs_norm.get_shape())
    return imgs_norm, imgs_raw, tags, class_labels

  def _create_inputs_for_pair(self, batch_size):
    assert self.group_size == 2
    assert batch_size % self.group_size == 0
    batch_size /= self.group_size

    def _create_example(_=None):
      rand = tf.random_uniform([5], maxval=tf.int32.max, dtype=tf.int32)
      sample_same_person = rand[0] % (self.pair_ratio + 1)
      sample_same_person = tf.cast(tf.equal(sample_same_person, 0), tf.int32)
      pers_id_1 = ((rand[1] - 1) % self.num_train_id) + 1
      pers_1_n_imgs = self.train_counts[pers_id_1 - 1]
      img_id_1 = ((rand[2] - 1) % pers_1_n_imgs) + 1

      def if_same_person():
        pers_id_2 = pers_id_1
        img_id_2 = ((rand[4] - 1) % (pers_1_n_imgs - 1)) + 1
        img_id_2 = tf.cond(img_id_2 >= img_id_1, lambda: img_id_2 + 1, lambda: img_id_2)
        return pers_id_2, img_id_2

      def if_not_same_person():
        pers_id_2 = ((rand[3] - 1) % (self.num_train_id - 1)) + 1
        pers_id_2 = tf.cond(pers_id_2 >= pers_id_1, lambda: pers_id_2 + 1, lambda: pers_id_2)
        pers_2_n_imgs = self.train_counts[pers_id_2 - 1]
        img_id_2 = ((rand[4] - 1) % pers_2_n_imgs) + 1
        return pers_id_2, img_id_2

      pers_id_2, img_id_2 = tf.cond(tf.cast(sample_same_person, tf.bool), if_same_person, if_not_same_person)

      #TODO: change the functions above to create 0 indexed values
      img_id_1 -= 1
      img_id_2 -= 1
      pers_id_1 -= 1
      pers_id_2 -= 1
      img1, img1_class, img1_file_name, original_img1 = self.load_crop(img_id_1, pers_id_1)
      img2, img2_class, img2_file_name, original_img2 = self.load_crop(img_id_2, pers_id_2)
      tag = img1_file_name + " " + img2_file_name + " " + tf.as_string(sample_same_person)
      pair = tf.stack([img1, img2])
      original_class_pair = tf.stack([img1_class, img2_class], axis=0)
      label = sample_same_person
      return pair, label, tag, original_class_pair

    USE_DATASET_API = False
    if USE_DATASET_API:
      dummy = tf.constant(0, tf.int32)
      dataset = tf.contrib.data.Dataset.from_tensors(dummy)
      dataset = dataset.map(_create_example)
      dataset = dataset.repeat()
      dataset = dataset.batch(batch_size)
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()
      imgs, labels, tags, original_classes = next_element
    else:
      pair, label, tag, original_class = _create_example()
      imgs, labels, tags, original_classes = tf.train.batch([pair, label, tag, original_class], batch_size=batch_size,
                                                            num_threads=32, capacity=50 * batch_size)

    imgs = self.reshape_group(imgs, batch_size)
    labels = tf.reshape(tf.tile(tf.expand_dims(labels, axis=-1), multiples=[1, 2]), [-1])
    return imgs, labels, tags, original_classes

  def _create_inputs_for_group(self, batch_size):

    assert 1 < self.group_size < batch_size
    assert batch_size % self.group_size == 0
    batch_size /= self.group_size

    batch_size = int(batch_size)

    sample_from = tf.range(self.num_train_id)
    if batch_size > self.num_train_id:
      sample_from = tf.tile(sample_from, [int(ceil(float(batch_size) / self.num_train_id))])
    pers_ids = tf.random_shuffle(sample_from)[0:int(batch_size)]

    def for_each_identity(p_idx):
      pers_id = pers_ids[p_idx]
      img_ids = tf.tile(tf.random_shuffle(tf.range(self.train_counts[pers_id])), [4])[:self.group_size]

      def for_each_img(i_idx):
        img_id = img_ids[i_idx]
        img, img_class, img_file_name, original_img = self.load_crop(img_id, pers_id)
        return img, img_class, img_file_name, original_img

      imgs, labels, tags, original_imgs = tf.map_fn(for_each_img, tf.range(self.group_size), dtype=(tf.float32, tf.int32, tf.string, tf.float32))
      return imgs, labels, tags, original_imgs

    imgs, labels, tags, original_imgs = tf.map_fn(for_each_identity, tf.range(batch_size), dtype=(tf.float32, tf.int32, tf.string, tf.float32))

    # imgs, labels, tags = tf.train.batch([imgs, labels, tags], batch_size=1, num_threads=32, capacity=50 * batch_size)
    # imgs, labels, tags = tf.train.batch([imgs, labels, tags], batch_size=1, num_threads=1, capacity=1 * batch_size)
    # imgs = self.reshape_group(tf.squeeze(imgs, 0), batch_size)
    # labels = self.reshape_group(tf.squeeze(labels, 0), batch_size)
    # tags = self.reshape_group(tf.squeeze(tags, 0), batch_size)

    imgs = self.reshape_group(imgs, batch_size)
    labels = self.reshape_group(labels, batch_size)
    tags = self.reshape_group(tags, batch_size)
    original_imgs = self.reshape_group(original_imgs,batch_size)
    return imgs, labels, tags, original_imgs

  def create_input_tensors_dict(self, batch_size):
    if self.batching_mode == "pair":
      imgs, labels, tags, original_classes = self._create_inputs_for_pair(batch_size)
      imgs_raw = unnormalize(imgs)
    elif self.batching_mode == "group":
      imgs, labels, tags, original_images = self._create_inputs_for_group(batch_size)
      original_classes = labels
      imgs_raw = original_images
      # imgs_raw = None
    elif self.batching_mode == "eval":
      imgs, imgs_raw, tags, labels = self._create_inputs_for_eval(batch_size)
      original_classes = labels
    else:
      raise ValueError("Incorrect batching mode error")

    #summary = tf.get_collection(tf.GraphKeys.SUMMARIES)[-1]
    #self.summaries.append(summary)
    if self.use_summaries:
      summ = tf.summary.image("imgs", unnormalize(imgs))
      self.summaries.append(summ)
    tensors = {"inputs": imgs, "labels": labels, "tags": tags, "original_labels": original_classes}
    if imgs_raw is not None:
      tensors["imgs_raw"] = imgs_raw
    return tensors

  def load_crop(self, img_id, pers_id):
    def select_data(pers_id_, img_id_):
      idx = self.indices_for_classes[pers_id_][img_id_]
      return self.class_labels_list[idx], self.bboxes_list[idx], self.file_names_list[idx]
    img_class, img_bbox, img_file_name = tf.py_func(select_data, [pers_id, img_id],
                                                    [tf.int32, tf.float32, tf.string], name="select_data")
    img_class.set_shape(())
    img_bbox.set_shape((4,))
    img_file_name.set_shape(())
    img, original_img, _ = self._load_crop_helper(img_file_name, img_bbox)
    return img, img_class, img_file_name,original_img

  def _load_crop_helper(self, img_file_name, img_bbox):
    img_whole_im = load_image_tensorflow(img_file_name, jpg=self.jpg, channels=3)
    dims = tf.shape(img_whole_im)
    img_x = img_bbox[0]
    img_y = img_bbox[1]
    img_w = img_bbox[2]
    img_h = img_bbox[3]
    # add context region
    img_x -= 0.5 * img_w * (self.context_region_factor - 1.0)
    img_y -= 0.5 * img_h * (self.context_region_factor - 1.0)
    img_w *= self.context_region_factor
    img_h *= self.context_region_factor
    # round to integer coordinates
    img_x = tf.cast(tf.round(img_x), tf.int32)
    img_y = tf.cast(tf.round(img_y), tf.int32)
    img_w = tf.cast(tf.round(img_w), tf.int32)
    img_h = tf.cast(tf.round(img_h), tf.int32)
    # clip to image size
    img_x = tf.maximum(img_x, 0)
    img_y = tf.maximum(img_y, 0)
    img_excess_w = tf.maximum(img_x + img_w - dims[1], 0)
    img_excess_h = tf.maximum(img_y + img_h - dims[0], 0)
    img_w = img_w - img_excess_w
    img_h = img_h - img_excess_h
    # crop
    img_cropped = img_whole_im[img_y:img_y + img_h, img_x:img_x + img_w]
    # resize
    img = resize_image(img_cropped, self.input_size, True)
    img.set_shape(self.input_size + (3,))
    # augment and normalize
    tensors = {"unnormalized_img": img}
    tensors = apply_augmentors(tensors, self.augmentors)
    img = tensors["unnormalized_img"]
    img_norm = normalize(img)
    return img_norm, img, img_cropped

  def reshape_group(self, x, batch_size):
    shape = smart_shape(x)
    shape2 = shape[1:]
    shape2[0] = self.group_size * batch_size
    x = tf.reshape(x, shape2)
    return x

  def num_examples_per_epoch(self):
    return self.epoch_length

  def void_label(self):
    return None
