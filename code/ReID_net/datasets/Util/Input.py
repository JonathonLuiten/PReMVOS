import tensorflow as tf

import ReID_net.Constants as Constants
from ReID_net.datasets.Util.Normalization import normalize


def assemble_input_tensors(tensors, img_size=(None, None)):
  img_size = list(img_size)
  assert "unnormalized_img" in tensors
  assert "tag" in tensors

  img = tensors["unnormalized_img"]
  img = normalize(img)
  tag = tensors["tag"]
  if "label" in tensors:
    label = tensors["label"]
  else:
    label = None
  if "raw_label" in tensors:
    raw_label = tensors["raw_label"]
  else:
    raw_label = label
  concats = [img]
  n_input_channels = 3
  if "old_label" in tensors:
    old_label = tf.cast(tensors["old_label"], tf.float32)

    if Constants.OLD_LABEL_AS_DT in tensors and tensors[Constants.OLD_LABEL_AS_DT] is True:
      clip_value = tf.ones_like(old_label) * 255
      old_label = tf.where(tf.greater(old_label, 255), clip_value, old_label)
      old_label /= 255.0

    concats.append(old_label)
    n_input_channels += 1

  if Constants.DT_NEG in tensors:
    # Do not use the click channel as they can be deciphered from the distance transforms.
    u0 = tensors[Constants.DT_NEG]
    u0 = tf.cast(u0, tf.float32)
    clip_value = tf.ones_like(u0)*255
    u0 = tf.where(tf.greater(u0, 255), clip_value, u0)
    u0 /= 255.0
    concats.append(u0)
    n_input_channels += u0.get_shape().as_list()[-1]
  if Constants.DT_POS in tensors:
    u1 = tensors[Constants.DT_POS]
    u1 = tf.cast(u1, tf.float32)
    clip_value = tf.ones_like(u1) * 255
    u1 = tf.where(tf.greater(u1, 255), clip_value, u1 )
    u1 /= 255.0
    concats.append(u1)
    n_input_channels += u1.get_shape().as_list()[-1]
  if "flow_past" in tensors:
    concats.append(tensors["flow_past"])
    n_input_channels += 2
  if "flow_future" in tensors:
    concats.append(tensors["flow_future"])
    n_input_channels += 2
  if len(concats) > 1:
    img = tf.concat(concats, axis=2)

  img.set_shape(img_size + [n_input_channels])

  tensors_out = {"inputs": img, "tags": tag}
  if label is not None:
    tensors_out["labels"] = label
  if raw_label is not None:
    tensors_out["raw_labels"] = raw_label
  if "index_img" in tensors:
    tensors_out["index_imgs"] = tensors["index_img"]
  #TODO: maybe just start with a copy of tensors instead of doing every case separately?
  if Constants.BBOXES in tensors:
    tensors_out[Constants.BBOXES] = tensors[Constants.BBOXES]
  if Constants.IDS in tensors:
    tensors_out[Constants.IDS] = tensors[Constants.IDS]
  if Constants.CLASSES in tensors:
    tensors_out[Constants.CLASSES] = tensors[Constants.CLASSES]
  if Constants.IMG_IDS in tensors:
    tensors_out[Constants.IMG_IDS] = tensors[Constants.IMG_IDS]
  if Constants.ORIGINAL_SIZES in tensors:
    tensors_out[Constants.ORIGINAL_SIZES] = tensors[Constants.ORIGINAL_SIZES]
  if Constants.RESIZED_SIZES in tensors:
    tensors_out[Constants.RESIZED_SIZES] = tensors[Constants.RESIZED_SIZES]
  if Constants.IGNORE_REGIONS in tensors:
    tensors_out[Constants.IGNORE_REGIONS] = tensors[Constants.IGNORE_REGIONS]
  if Constants.SCENE_INFOS in tensors:
    tensors_out[Constants.SCENE_INFOS] = tensors[Constants.SCENE_INFOS]
  if Constants.OLD_LABEL_AS_DT in tensors:
    tensors_out[Constants.OLD_LABEL_AS_DT] = tensors[Constants.OLD_LABEL_AS_DT]
  return tensors_out
