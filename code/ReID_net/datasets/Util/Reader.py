import numpy as np
import tensorflow as tf

import ReID_net.Constants as Constants
from ReID_net.datasets.Augmentors import apply_augmentors
from ReID_net.datasets.Util import Util
from ReID_net.datasets.Util.Input import assemble_input_tensors
from ReID_net.datasets.Util.Normalization import normalize
from ReID_net.datasets.Util.Resize import resize
from ReID_net.datasets.Util.Util import load_flow_from_flo, create_index_image, smart_shape


def create_tensor_dict(unnormalized_img, label, tag, raw_label=None, old_label=None, flow_past=None, flow_future=None,
                       use_index_img=False, u0=None, u1=None, bboxes=None, ids=None, classes=None, img_id=None,
                       ignore_regions=None, scene_infos=None, old_label_as_dt = None):
  tensors = {"unnormalized_img": unnormalized_img, "tag": tag}
  #note that "label" is a bit a misnomer, since it's actually a mask. better rename it at some point
  if label is not None:
    tensors["label"] = label
  if raw_label is None:
    if label is not None:
      tensors["raw_label"] = label
  else:
    tensors["raw_label"] = raw_label
  if old_label is not None:
    tensors["old_label"] = old_label
  if flow_past is not None:
    tensors["flow_past"] = flow_past
  if flow_future is not None:
    tensors["flow_future"] = flow_future
  if bboxes is not None:
    tensors[Constants.BBOXES] = bboxes
  if ids is not None:
    tensors[Constants.IDS] = ids
  if classes is not None:
    tensors[Constants.CLASSES] = classes
  if u0 is not None:
    tensors[Constants.DT_NEG] = u0
  if u1 is not None:
    tensors[Constants.DT_POS] = u1
  if img_id is not None:
    tensors[Constants.IMG_IDS] = img_id
  if ignore_regions is not None:
    tensors[Constants.IGNORE_REGIONS] = ignore_regions
  if scene_infos is not None:
    tensors[Constants.SCENE_INFOS] = scene_infos
  if old_label_as_dt is not None:
    tensors[Constants.OLD_LABEL_AS_DT] = old_label_as_dt
  if use_index_img:
    shape = smart_shape(unnormalized_img)
    index_img = create_index_image(shape[0], shape[1])
    tensors["index_img"] = index_img
  return tensors


def load_label_default(img_path, label_path, channels=1):
  label_contents = tf.read_file(label_path)
  label = tf.image.decode_image(label_contents, channels=channels)
  labels = {"label": label}
  return labels


def load_img_default(img_path):
  img_contents = tf.read_file(img_path)

  img = tf.image.decode_image(img_contents, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img.set_shape((None, None, 3))

  return img


def read_images_from_disk(input_queue, input_size, resize_mode, label_postproc_fn=lambda x: x, augmentors=(),
                          label_load_fn=load_label_default, img_load_fn=load_img_default,
                          distance_transform_fn=Util.create_distance_transform):
  im_path = input_queue[0]
  label_path = input_queue[1]
  if len(input_queue) > 2:
    img_id = input_queue[2]
  else:
    img_id = None
  img = img_load_fn(img_path=im_path)

  #TODO: clean up all this stuff!
  labels = label_load_fn(im_path, label_path)
  if 'label' in list(labels.keys()):
    label = labels['label']
    label = label_postproc_fn(label)
    label.set_shape(img.get_shape().as_list()[:-1] + [1])
  else:
    label = None
  if 'old_label' in list(labels.keys()):
    old_label = labels['old_label']
    old_label.set_shape(img.get_shape().as_list()[:-1] + [1])
  else:
    old_label = None
  if Constants.BBOXES in list(labels.keys()):
    bboxes = labels[Constants.BBOXES]
  else:
    bboxes = None
  if Constants.IDS in list(labels.keys()):
    ids = labels[Constants.IDS]
  else:
    ids = None
  if Constants.CLASSES in list(labels.keys()):
    classes = labels[Constants.CLASSES]
  else:
    classes = None
  if Constants.IGNORE_REGIONS in list(labels.keys()):
    ignore_regions = labels[Constants.IGNORE_REGIONS]
  else:
    ignore_regions = None
  if Constants.SCENE_INFOS in list(labels.keys()):
    scene_infos = labels[Constants.SCENE_INFOS]
  else:
    scene_infos = None
  if Constants.OLD_LABEL_AS_DT in list(labels.keys()):
    old_label_as_dt = labels[Constants.OLD_LABEL_AS_DT]
  else:
    old_label_as_dt = None
  u0 = None
  u1 = None

  tensors = create_tensor_dict(unnormalized_img=img, label=label, old_label=old_label, u0=u0, u1=u1, tag=im_path,
                               raw_label=label, bboxes=bboxes, ids=ids, classes=classes, img_id=img_id,
                               ignore_regions=ignore_regions, scene_infos=scene_infos, old_label_as_dt = old_label_as_dt)

  tensors = resize(tensors, resize_mode, input_size)

  # Create distance transform after image resize to speed up the computation.
  if Constants.USE_CLICKS in list(labels.keys()):
    assert Constants.STRATEGY in labels and Constants.IGNORE_CLASSES in labels
    tensors = add_distance_transform(tensors, labels, distance_transform_fn)
  elif Constants.OLD_LABEL_AS_DT in list(labels.keys()):
    tensors["old_label"] = tf.py_func(distance_transform_fn, [tensors["label"]], [tf.float32])[0]
    tensors["old_label"].set_shape(tensors["label"].get_shape())

  tensors = apply_augmentors(tensors, augmentors)
  tensors = assemble_input_tensors(tensors)

  summaries = []

  return tensors, summaries


def create_clicks_map(clicks, dt):
  click_map = np.zeros_like(dt)
  if clicks.shape[0] > 0:
    click_map[clicks[:, 0], clicks[:, 1]] = 1

  return click_map.astype(np.float32)


def load_image_tensorflow(im_path, jpg, channels=None):
  img_contents = tf.read_file(im_path)
  if jpg:
    img = tf.image.decode_jpeg(img_contents, channels=channels)
  else:
    img = tf.image.decode_png(img_contents, channels=channels)
  #for some reason, using decode_image makes things slow!
  #img = tf.image.decode_image(img_contents)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img.set_shape([None, None, None])
  return img


def load_normalized_image_tensorflow(im_path, jpg):
  img = load_image_tensorflow(im_path, jpg=jpg)
  img = normalize(img)
  return img


def load_png_mask_tensorflow(path, divide_by_255=True):
  contents = tf.read_file(path)
  mask = tf.image.decode_png(contents, channels=1)
  mask = tf.cast(mask, tf.float32)
  if divide_by_255:
    mask /= 255
  return mask


def load_flow_from_flo_tensorflow(fn, flow_as_angle):
  def my_load_flow(f):
    return load_flow_from_flo(f, flow_as_angle)
  flow, = tf.py_func(my_load_flow, [fn], [tf.float32])
  return flow


def add_distance_transform(tensors, labels, distance_transform_fn):
  args_list = [tensors["unnormalized_img"], tensors["label"],
               tensors["raw_label"], labels[Constants.STRATEGY], labels[Constants.IGNORE_CLASSES]]

  if "old_label" in tensors:
    args_list.append(tensors["old_label"])

  u0, u1, num_clicks = tf.py_func(distance_transform_fn,
                                  args_list,
                                  [tf.float32, tf.float32, tf.int64],
                                  name="create_distance_transform")

  u0 = tf.expand_dims(u0, axis=2)
  u0.set_shape(tensors["unnormalized_img"].get_shape().as_list()[:-1] + [1])

  u1 = tf.expand_dims(u1, axis=2)
  u1.set_shape(tensors["unnormalized_img"].get_shape().as_list()[:-1] + [1])

  shape = tensors["tag"].get_shape()
  im_path = tf.string_join([tensors["tag"], tf.as_string(num_clicks)], separator=":", name="JoinPath")
  im_path.set_shape(shape)

  tensors[Constants.DT_NEG] = u0
  tensors[Constants.DT_POS] = u1
  tensors["tag"] = im_path

  return tensors
