import tensorflow as tf

import ReID_net.Constants as Constants
from ReID_net.datasets.Util.Util import smart_shape

# enum
class ResizeMode:
  def __init__(self):
    pass

  RandomCrop, ResizeShorterSideToFixedSize, Unchanged, FixedSize, ResizeAndCrop, RandomResizeAndCrop, \
  RandomResizeAndObjectCrop, RandomResize, RandomResizeLess, DetectionFixedSize, \
  DetectionFixedSizeForEval, DetectionKeepAspectRatio, DetectionKeepAspectRatioForEval, \
  RandomResizeHalfSize, RandomResizeAndFixedObjectCrop, BboxResizeAndCrop = list(range(16))


def parse_resize_mode(mode):
  mode = mode.lower()
  if mode == "random_crop":
    return ResizeMode.RandomCrop
  elif mode == "resize_shorter_side_to_fixed_size":
    return ResizeMode.ResizeShorterSideToFixedSize
  elif mode == "unchanged":
    return ResizeMode.Unchanged
  elif mode == "fixed_size":
    return ResizeMode.FixedSize
  elif mode == "resize_and_crop":
    return ResizeMode.ResizeAndCrop
  elif mode == "random_resize_and_crop":
    return ResizeMode.RandomResizeAndCrop
  elif mode == "random_resize_and_object_crop":
    return ResizeMode.RandomResizeAndObjectCrop
  elif mode == "random_resize_and_fixed_object_crop":
    return ResizeMode.RandomResizeAndFixedObjectCrop
  elif mode == "random_resize":
    return ResizeMode.RandomResize
  elif mode == "random_resize_half_size":
    return ResizeMode.RandomResizeHalfSize
  elif mode == "random_resize_less":
    return ResizeMode.RandomResizeLess
  elif mode == "detection_fixed_size":
    return ResizeMode.DetectionFixedSize
  elif mode == "detection_fixed_size_for_eval":
    return ResizeMode.DetectionFixedSizeForEval
  elif mode == "detection_keep_aspect_ratio":
    return ResizeMode.DetectionKeepAspectRatio
  elif mode == "detection_keep_aspect_ratio_for_eval":
    return ResizeMode.DetectionKeepAspectRatioForEval
  elif mode == "bbox_crop_and_resize":
    return ResizeMode.BboxResizeAndCrop
  else:
    assert False, "Unknown resize mode " + mode


def resize_image(img, out_size, bilinear):
  if bilinear:
    img = tf.image.resize_images(img, out_size)
  else:
    img = tf.image.resize_nearest_neighbor(tf.expand_dims(img, 0), out_size)
    img = tf.squeeze(img, 0)
  return img


# adapted from code from tf.random_crop
def random_crop_image(img, size, offset=None):
  shape = tf.shape(img)
  #remove the assertion for now since it makes the queue filling slow for some reason
  #check = tf.Assert(
  #  tf.reduce_all(shape[:2] >= size),
  #  ["Need value.shape >= size, got ", shape, size])
  #with tf.control_dependencies([check]):
  #  img = tf.identity(img)
  limit = shape[:2] - size + 1
  dtype = tf.int32
  if offset is None:
    offset = tf.random_uniform(shape=(2,), dtype=dtype, maxval=dtype.max, seed=None) % limit
    offset = tf.stack([offset[0], offset[1], 0])
  size0 = size[0] if isinstance(size[0], int) else None
  size1 = size[1] if isinstance(size[1], int) else None
  size_im = tf.stack([size[0], size[1], img.get_shape().as_list()[2]])
  img_cropped = tf.slice(img, offset, size_im)
  out_shape_img = [size0, size1, img.get_shape()[2]]
  img_cropped.set_shape(out_shape_img)
  return img_cropped, offset


def random_crop_tensors(tensors, size):
  tensors_cropped = tensors.copy()
  tensors_cropped["unnormalized_img"], offset = random_crop_image(tensors["unnormalized_img"], size)
  tensors_cropped["label"], offset = random_crop_image(tensors["label"], size, offset)
  tensors_cropped["raw_label"] = tensors_cropped["label"]
  if "old_label" in tensors:
    tensors_cropped["old_label"], offset = random_crop_image(tensors["old_label"], size, offset)
  if Constants.DT_NEG in tensors:
    tensors_cropped[Constants.DT_NEG], offset = random_crop_image(tensors[Constants.DT_NEG], size, offset)
  if Constants.DT_POS in tensors:
    tensors_cropped[Constants.DT_POS], offset = random_crop_image(tensors[Constants.DT_POS], size, offset)

  return tensors_cropped


def object_crop(tensors, size):
  tensors_cropped = tensors.copy()
  label = tensors["label"]
  object_locations = tf.cast(tf.where(tf.not_equal(label, 0))[:, :2], tf.int32)
  shape = tf.shape(label)

  min_val = tf.maximum(tf.constant([0, 0]),
                       tf.reduce_max(object_locations, axis=0) - size + 50)
  max_val = tf.minimum(tf.reduce_min(object_locations, axis=0) - 50, shape[:2] - size + 1)
  max_val = tf.where(tf.greater_equal(min_val, max_val), min_val+1, max_val)

  offset_1 = tf.random_uniform(shape=(), dtype=tf.int32, minval=min_val[0], maxval=max_val[0], seed=None)
  offset_2 = tf.random_uniform(shape=(), dtype=tf.int32, minval=min_val[1], maxval=max_val[1], seed=None)
  offset = tf.stack([offset_1, offset_2, 0])

  tensors_cropped["unnormalized_img"], offset = random_crop_image(tensors["unnormalized_img"], size, offset=offset)
  tensors_cropped["label"], offset = random_crop_image(tensors["label"], size, offset)
  tensors_cropped["raw_label"] = tensors_cropped["label"]
  if "old_label" in tensors:
    tensors_cropped["old_label"], offset = random_crop_image(tensors["old_label"], size, offset)
  if Constants.DT_NEG in tensors:
    tensors_cropped[Constants.DT_NEG], offset = random_crop_image(tensors[Constants.DT_NEG], size, offset)
  if Constants.DT_POS in tensors:
    tensors_cropped[Constants.DT_POS], offset = random_crop_image(tensors[Constants.DT_POS], size, offset)

  return tensors_cropped


def bbox_resize_and_crop(tensors, size):
  MARGIN = 50
  tensors_cropped = tensors.copy()
  label = tensors["label"]
  object_locations = tf.cast(tf.where(tf.not_equal(label, 0))[:, :2], tf.int32)
  shape = tf.shape(tensors["unnormalized_img"])

  min_row = tf.maximum(tf.reduce_min(object_locations[:, 0]) - MARGIN, 0)
  min_col = tf.maximum(tf.reduce_min(object_locations[:, 1]) - MARGIN, 0)

  max_row = tf.minimum(tf.reduce_max(object_locations[:, 0]) + MARGIN, shape[0])
  max_col = tf.minimum(tf.reduce_max(object_locations[:, 1]) + MARGIN, shape[1])

  label_resized = label[min_row: max_row,
                        min_col: max_col]
  tensors_cropped["label"] = resize_image(label_resized, size, bilinear=False)

  img_resized = tensors["unnormalized_img"][min_row: max_row, min_col: max_col]
  tensors_cropped["unnormalized_img"] = resize_image(img_resized, size, bilinear=True)
  if "old_label" in tensors and Constants.OLD_LABEL_AS_DT not in tensors:
    old_label = tensors["old_label"][min_row: max_row, min_col: max_col]
    tensors_cropped["old_label"] = resize_image(old_label, size, bilinear=False)

  tensors_cropped["raw_label"] = tensors_cropped["label"]

  return tensors_cropped


def object_crop_fixed_offset(tensors, size):
  tensors_cropped = tensors.copy()
  label = tensors["label"]
  object_locations = tf.cast(tf.where(tf.not_equal(label, 0))[:, :2], tf.int32)
  shape = tf.shape(label)

  min_val = tf.maximum(tf.constant([0, 0]),
                       tf.reduce_max(object_locations, axis=0) - size)
  offset = tf.concat([min_val, [0]], axis=0)

  tensors_cropped["unnormalized_img"], offset = random_crop_image(tensors["unnormalized_img"], size, offset=offset)
  tensors_cropped["label"], offset = random_crop_image(tensors["label"], size, offset)
  tensors_cropped["raw_label"] = tensors_cropped["label"]
  if "old_label" in tensors:
    tensors_cropped["old_label"], offset = random_crop_image(tensors["old_label"], size, offset)
  if Constants.DT_NEG in tensors:
    tensors_cropped[Constants.DT_NEG], offset = random_crop_image(tensors[Constants.DT_NEG], size, offset)
  if Constants.DT_POS in tensors:
    tensors_cropped[Constants.DT_POS], offset = random_crop_image(tensors[Constants.DT_POS], size, offset)

  return tensors_cropped


def resize(tensors, resize_mode, input_size):
  tensors = tensors.copy()
  if resize_mode == ResizeMode.RandomCrop:
    tensors = random_crop_tensors(tensors, input_size)
  elif resize_mode == ResizeMode.ResizeShorterSideToFixedSize:
    assert len(input_size) == 1
    tensors = resize_shorter_side_fixed_size(tensors, input_size[0])
  elif resize_mode == ResizeMode.Unchanged:
    tensors = resize_unchanged(tensors, input_size)
  elif resize_mode == ResizeMode.FixedSize:
    tensors = resize_fixed_size(tensors, input_size)
  elif resize_mode == ResizeMode.ResizeAndCrop:
    assert len(input_size) == 3
    tensors = resize_shorter_side_fixed_size(tensors, input_size[0])
    tensors = random_crop_tensors(tensors, input_size[1:])
  elif resize_mode == ResizeMode.RandomResizeAndCrop:
    assert len(input_size) in (1, 2)
    if len(input_size) == 2:
      assert input_size[0] == input_size[1]
      crop_size = input_size
    else:
      crop_size = [input_size, input_size]
    tensors = resize_random_scale_with_min_size(tensors, min_size=crop_size)
    tensors = random_crop_tensors(tensors, crop_size)
  elif resize_mode == ResizeMode.RandomResizeAndObjectCrop:
    assert len(input_size) in (1, 2)
    if len(input_size) == 2:
      assert input_size[0] == input_size[1]
      crop_size = input_size
    else:
      crop_size = [input_size, input_size]
    tensors = resize_random_scale_with_min_size(tensors, min_size=crop_size)
    tensors = object_crop(tensors, crop_size)
  elif resize_mode == ResizeMode.RandomResizeAndFixedObjectCrop:
    assert len(input_size) in (1, 2)
    if len(input_size) == 2:
      assert input_size[0] == input_size[1]
      crop_size = input_size
    else:
      crop_size = [input_size, input_size]
    tensors = scale_with_min_size(tensors, min_size=crop_size)
    tensors = object_crop_fixed_offset(tensors, crop_size)
  elif resize_mode == ResizeMode.RandomResize:
    tensors = resize_random_scale_with_min_size(tensors, min_size=(0, 0))
  elif resize_mode == ResizeMode.RandomResizeHalfSize:
    min_scale = 0.35
    max_scale = 0.65
    tensors = resize_random_scale_with_min_size(tensors, min_size=(0, 0), min_scale=min_scale, max_scale=max_scale)
  elif resize_mode == ResizeMode.RandomResizeLess:
    tensors = resize_random_scale_with_min_size(tensors, min_size=(0, 0), min_scale=0.85, max_scale=1.15)
  elif resize_mode == ResizeMode.DetectionFixedSize:
    assert len(input_size) in (1, 2)
    if len(input_size) == 1:
      input_size = [input_size, input_size]
    tensors = resize_detection_fixed_size(tensors, input_size)
  elif resize_mode == ResizeMode.DetectionFixedSizeForEval:
    assert len(input_size) in (1, 2)
    if len(input_size) == 1:
      input_size = [input_size, input_size]
    tensors = resize_detection_fixed_size(tensors, input_size, for_testing=True)
  elif resize_mode == ResizeMode.DetectionKeepAspectRatio:
    assert len(input_size) == 2
    tensors = resize_detection_keep_aspect_ratio(tensors, input_size, for_testing=False)
  elif resize_mode == ResizeMode.DetectionKeepAspectRatioForEval:
    assert len(input_size) == 2
    tensors = resize_detection_keep_aspect_ratio(tensors, input_size, for_testing=True)
  elif resize_mode == ResizeMode.BboxResizeAndCrop:
    assert len(input_size) == 2
    tensors = bbox_resize_and_crop(tensors, input_size)
  else:
    assert False, ("Unknown resize mode", resize_mode)
  return tensors


def resize_random_scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors["unnormalized_img"]

  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  shorter_side = tf.minimum(h, w)
  min_scale_factor = tf.cast(min_size, tf.float32) / tf.cast(shorter_side, tf.float32)
  min_scale = tf.maximum(min_scale, min_scale_factor)
  max_scale = tf.maximum(max_scale, min_scale_factor)
  scale_factor = tf.random_uniform(shape=[], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  scaled_size = tf.cast(tf.round(tf.cast(tf.shape(img)[:2], tf.float32) * scale_factor), tf.int32)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def scale_with_min_size(tensors, min_size, min_scale=0.7, max_scale=1.3):
  assert min_size is not None
  img = tensors["unnormalized_img"]

  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  shorter_side = tf.minimum(h, w)
  min_scale_factor = tf.cast(min_size, tf.float32) / tf.cast(shorter_side, tf.float32)
  scaled_size = tf.cast(tf.round(tf.cast(tf.shape(img)[:2], tf.float32) * min_scale_factor), tf.int32)
  tensors_out = resize_fixed_size(tensors, scaled_size)
  return tensors_out


def resize_fixed_size(tensors, input_size):
  tensors_out = tensors.copy()
  assert input_size is not None
  img = tensors["unnormalized_img"]
  img = resize_image(img, input_size, True)
  if "old_label" in tensors:
    old_label = tensors["old_label"]
    old_label = resize_image(old_label, input_size, False)
    tensors_out["old_label"] = old_label
  if Constants.DT_NEG in tensors:
    u0 = tensors[Constants.DT_NEG]
    u0 = resize_image(u0, input_size, False)
    tensors_out[Constants.DT_NEG] = u0
  if Constants.DT_POS in tensors:
    u1 = tensors[Constants.DT_POS]
    u1 = resize_image(u1, input_size, False)
    tensors_out[Constants.DT_POS] = u1
    print("Shape of u1: " + repr(u1.get_shape()))
  tensors_out["unnormalized_img"] = img
  if "label" in tensors:
    label = tensors["label"]
    label = resize_image(label, input_size, False)
    tensors_out["label"] = label

  #do not change raw_label
  #TODO: this behaviour is different to previous version, check if it breaks anything
  #tensors_out["raw_label"] = label  # raw_label refers to after resizing but before augmentations
  return tensors_out


def resize_shorter_side_fixed_size(tensors, input_size):
  assert input_size is not None
  img = tensors["unnormalized_img"]
  h = tf.shape(img)[0]
  w = tf.shape(img)[1]
  size = tf.constant(input_size)
  h_new = tf.cond(h < w, lambda: size, lambda: h * size / w)
  w_new = tf.cond(h < w, lambda: w * size / h, lambda: size)
  new_shape = tf.stack([h_new, w_new])
  tensors_out = resize_fixed_size(tensors, new_shape)
  return tensors_out


def resize_unchanged(tensors, input_size):
  tensors_out = tensors.copy()
  if input_size is not None:
    def _set_shape(key, n_channels=None):
      if key in tensors:
        tensor = tensors[key]
        tensor.set_shape((input_size[0], input_size[1], n_channels))

    _set_shape(Constants.UNNORMALIZED_IMG)
    _set_shape("label")
    _set_shape("old_label", 1)
    if Constants.DT_NEG in tensors:
      _set_shape(Constants.DT_NEG, tensors[Constants.DT_NEG].get_shape().as_list()[-1])
    if Constants.DT_POS in tensors:
      _set_shape(Constants.DT_POS, tensors[Constants.DT_POS].get_shape().as_list()[-1])
    _set_shape("flow_future", 2)
    _set_shape("flow_past", 2)
    _set_shape("index_img", 2)
  return tensors_out


def resize_detection_fixed_size(tensors, input_size, for_testing=False):
  tensors_out = tensors.copy()
  #ignore_regions are currently not supported in this resize mode
  if Constants.IGNORE_REGIONS in tensors_out:
    del tensors_out[Constants.IGNORE_REGIONS]
  img = tensors[Constants.UNNORMALIZED_IMG]
  original_img = img
  bboxes = tensors[Constants.BBOXES]

  # remove the padding
  n_real_detections = tf.reduce_sum(tf.cast(tensors[Constants.IDS] > 0, tf.int32))
  bboxes = bboxes[:n_real_detections]
  classes = tensors[Constants.CLASSES][:n_real_detections]

  # permute y1, y2, x1, x2 -> y1, x1, y2, x1
  bboxes = tf.stack([bboxes[..., 0], bboxes[..., 2], bboxes[..., 1], bboxes[..., 3]], axis=-1)

  # normalize bboxes to [0..1]
  height = tf.shape(img)[0]
  width = tf.shape(img)[1]
  bboxes = tf.cast(bboxes, tf.float32) / tf.cast(tf.stack([height, width, height, width], axis=0), tf.float32)

  import object_detection.core.preprocessor as preproc
  if not for_testing:
    #crop (ssd style)
    img, bboxes, classes = preproc.ssd_random_crop(img, bboxes, classes)
    #alternative
    #img, bboxes, classes = preproc.random_crop_image(img, real_boxes, real_boxes)

    # include random horizontal flip augmentation here
    img, bboxes = preproc.random_horizontal_flip(img, bboxes)

  #resize image, note: boxes don't need resizing as they are in relative coordinates
  img = preproc.resize_image(img, new_height=input_size[0], new_width=input_size[1])

  if for_testing:
    _, bboxes = preproc.scale_boxes_to_pixel_coordinates(original_img, bboxes)
  else:
    _, bboxes = preproc.scale_boxes_to_pixel_coordinates(img, bboxes)

  #permute back y1, x1, y2, x1 -> y1, y2, x1, x2
  bboxes = tf.stack([bboxes[..., 0], bboxes[..., 2], bboxes[..., 1], bboxes[..., 3]], axis=-1)

  #pad the stuff needs to be padded back to the maximum size
  padded_size = smart_shape(tensors[Constants.CLASSES])[0]
  n_real_detections_after_crop = smart_shape(bboxes)[0]
  pad_size = padded_size - n_real_detections_after_crop
  paddings_bboxes = [[0, pad_size], [0, 0]]
  bboxes = tf.pad(bboxes, paddings=paddings_bboxes)
  paddings_classes_ids = [[0, pad_size]]
  classes = tf.pad(classes, paddings=paddings_classes_ids)
  ids = tf.pad(tf.range(n_real_detections_after_crop) + 1, paddings=paddings_classes_ids)
  if isinstance(padded_size, int):
    bboxes.set_shape((padded_size, 4))
    classes.set_shape((padded_size,))
    ids.set_shape((padded_size,))
  else:
    bboxes.set_shape((None, 4))
  #note that we do not retain the original ids, but it does not matter since this resize_mode is only meant for
  #isolated frames
  tensors_out[Constants.UNNORMALIZED_IMG] = img
  tensors_out[Constants.BBOXES] = bboxes
  tensors_out[Constants.CLASSES] = classes
  tensors_out[Constants.IDS] = ids
  tensors_out[Constants.RESIZED_SIZES] = tf.shape(img)[:2]
  if for_testing:
    tensors_out[Constants.ORIGINAL_SIZES] = tf.shape(original_img)[:2]
  return tensors_out


def resize_detection_keep_aspect_ratio(tensors, input_size, for_testing):
  tensors_out = tensors.copy()
  img = tensors[Constants.UNNORMALIZED_IMG]
  original_img = img
  bboxes = tensors[Constants.BBOXES]
  if Constants.IGNORE_REGIONS in tensors:
    ignore_regions = tensors[Constants.IGNORE_REGIONS]
  else:
    ignore_regions = None
  if Constants.SCENE_INFOS in tensors:
    scene_infos = tensors[Constants.SCENE_INFOS]
  else:
    scene_infos = None

  # permute y1, y2, x1, x2 -> y1, x1, y2, x1
  bboxes = tf.stack([bboxes[..., 0], bboxes[..., 2], bboxes[..., 1], bboxes[..., 3]], axis=-1)
  if ignore_regions is not None:
    ignore_regions = tf.stack([ignore_regions[..., 0], ignore_regions[..., 2], ignore_regions[..., 1],
                               ignore_regions[..., 3]], axis=-1)

  # normalize bboxes to [0..1]
  height = tf.shape(img)[0]
  width = tf.shape(img)[1]
  bboxes = tf.cast(bboxes, tf.float32) / tf.cast(tf.stack([height, width, height, width], axis=0), tf.float32)
  if ignore_regions is not None:
    ignore_regions = tf.cast(ignore_regions, tf.float32) / tf.cast(tf.stack([height, width, height, width],
                                                                            axis=0), tf.float32)

  import object_detection.core.preprocessor as preproc
  if not for_testing:
    if scene_infos is None:
      img, bboxes = preproc.random_horizontal_flip(img, bboxes)
      img = preproc.resize_to_range(img, min_dimension=input_size[0], max_dimension=input_size[1])
    else:
      #abuse masks argument for scene infos
      #temporarily switch the channel dimension (size 3) to number of masks dimension
      scene_infos = tf.transpose(scene_infos, [2, 0, 1])
      img, bboxes, scene_infos = preproc.random_horizontal_flip(img, bboxes, masks=scene_infos)
      img, scene_infos = preproc.resize_to_range(img, min_dimension=input_size[0], max_dimension=input_size[1],
                                                 masks=scene_infos)
      #undo shuffling of dimensions
      scene_infos = tf.transpose(scene_infos, [1, 2, 0])

  if for_testing:
    _, bboxes = preproc.scale_boxes_to_pixel_coordinates(original_img, bboxes)
    ignore_regions = None
  else:
    _, bboxes = preproc.scale_boxes_to_pixel_coordinates(img, bboxes)
    if ignore_regions is not None:
      _, ignore_regions = preproc.scale_boxes_to_pixel_coordinates(img, ignore_regions)

  # permute back y1, x1, y2, x1 -> y1, y2, x1, x2
  bboxes = tf.stack([bboxes[..., 0], bboxes[..., 2], bboxes[..., 1], bboxes[..., 3]], axis=-1)
  if ignore_regions is not None:
    ignore_regions = tf.stack([ignore_regions[..., 0], ignore_regions[..., 2], ignore_regions[..., 1],
                               ignore_regions[..., 3]], axis=-1)

  tensors_out[Constants.UNNORMALIZED_IMG] = img
  tensors_out[Constants.BBOXES] = bboxes
  tensors_out[Constants.RESIZED_SIZES] = tf.shape(img)[:2]
  if ignore_regions is not None:
    tensors_out[Constants.IGNORE_REGIONS] = ignore_regions
  if scene_infos is not None:
    tensors_out[Constants.SCENE_INFOS] = scene_infos
  if for_testing:
    tensors_out[Constants.ORIGINAL_SIZES] = tf.shape(original_img)[:2]
  return tensors_out
