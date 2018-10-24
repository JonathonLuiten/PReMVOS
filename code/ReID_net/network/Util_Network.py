import math
from collections import namedtuple

import numpy
import tensorflow as tf

TowerSetup = namedtuple("TowerSetup", ["dtype", "gpu", "is_main_train_tower", "is_training",
                                       "freeze_batchnorm", "variable_device", "use_update_ops_collection",
                                       "batch_size", "original_sizes", "resized_sizes", "use_weight_summaries"])


def conv2d(x, W, strides=None):
  if strides is None:
    strides = [1, 1]
  return tf.nn.conv2d(x, W, strides=[1] + strides + [1], padding="SAME")


def conv2d_dilated(x, W, dilation):
  res = tf.nn.atrous_conv2d(x, W, dilation, padding="SAME")
  shape = x.get_shape().as_list()
  shape[-1] = W.get_shape().as_list()[-1]
  res.set_shape(shape)
  return res


def max_pool(x, shape, strides=None):
  if strides is None:
    strides = shape
  return tf.nn.max_pool(x, ksize=[1] + shape + [1],
                        strides=[1] + strides + [1], padding="SAME")


def avg_pool(x, shape):
  return tf.nn.avg_pool(x, ksize=[1] + shape + [1],
                        strides=[1] + shape + [1], padding="VALID")
  #TODO: maywe be should change this to SAME


def global_avg_pool(x):
  assert len(x.get_shape()) == 4
  return tf.reduce_mean(x, [1, 2])


def apply_dropout(inp, dropout):
  if dropout == 0.0:
    return inp
  else:
    keep_prob = 1.0 - dropout
    return tf.nn.dropout(inp, keep_prob)


def prepare_input(inputs):
  #assert len(inputs) == 1, "Multiple inputs not yet implemented"
  if len(inputs) == 1:
    inp = inputs[0]
    dim = int(inp.get_shape()[-1])
  else:
    dims = [int(inp.get_shape()[3]) for inp in inputs]
    dim = sum(dims)
    inp = tf.concat_v2(inputs, 3)
  return inp, dim


def prepare_collapsed_input_and_dropout(inputs, dropout):
  assert len(inputs) == 1, "Multiple inputs not yet implemented"
  inp = inputs[0]
  shape = inp.get_shape()
  if len(shape) == 4:
    dim = int(numpy.prod(shape[1:4]))
    inp = tf.reshape(inp, [-1, dim])
  else:
    dim = int(shape[-1])
  if dropout != 0.0:
    keep_prob = 1.0 - dropout
    inp = tf.nn.dropout(inp, keep_prob)
  return inp, dim


activs = {"relu": tf.nn.relu, "linear": lambda x: x, "elu": tf.nn.elu}


def get_activation(act_str):
  assert act_str.lower() in activs, "Unknown activation function " + act_str
  return activs[act_str.lower()]


def create_batch_norm_vars(n_out, tower_setup, scope_name="bn"):
  dtype = tower_setup.dtype
  with tf.device(tower_setup.variable_device), tf.variable_scope(scope_name):
    initializer_zero = tf.constant_initializer(0.0, dtype=dtype)
    beta = tf.get_variable("beta", [n_out], dtype, initializer_zero)
    initializer_gamma = tf.constant_initializer(1.0, dtype=dtype)
    gamma = tf.get_variable("gamma", [n_out], dtype, initializer_gamma)
    mean_ema = tf.get_variable("mean_ema", [n_out], dtype, initializer_zero, trainable=False)
    var_ema = tf.get_variable("var_ema", [n_out], dtype, initializer_zero, trainable=False)
    return beta, gamma, mean_ema, var_ema


#adapted from https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
def create_bilinear_upsampling_weights(shape):
  height, width = shape[0], shape[1]
  f = math.ceil(width / 2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = numpy.zeros([shape[0], shape[1]])
  for x in range(width):
    for y in range(height):
      value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
      bilinear[x, y] = value
  weights = numpy.zeros(shape)
  for i in range(shape[2]):
    weights[:, :, i, i] = bilinear
  return weights


#adapted from Jakob Bauer
def iou_from_logits(logits, labels):
  """
  Computes the intersection over union (IoU) score for given logit tensor and target labels
  :param logits: 4D tensor of shape [batch_size, height, width, num_classes]
  :param labels: 3D tensor of shape [batch_size, height, width] and type int32 or int64
  :return: 1D tensor of shape [num_classes] with intersection over union for each class, averaged over batch
  """

  with tf.variable_scope("IoU"):
    # compute predictions
    preds = tf.arg_max(logits, dimension=3)

    num_labels = logits.get_shape().as_list()[-1]
    IoUs = []
    for label in range(num_labels):
      # find pixels with given label
      P = tf.equal(preds, label)
      L = tf.equal(labels, label)

      # Union
      U = tf.logical_or(P, L)
      U = tf.reduce_sum(tf.cast(U, tf.float32))

      # intersection
      I = tf.logical_and(P, L)
      I = tf.reduce_sum(tf.cast(I, tf.float32))

      IoUs.append(I / U)

    return tf.reshape(tf.stack(IoUs), (num_labels,))


def upsample_repeat(x, factor=2):
  #(batch, height, width, feat) -> (batch, height, 1, width, feat) -> (batch, height, 1, width, 1, feat)
  #-> (batch, height, 2, width, 2, feat) -> (batch, 2 * height, 2 * width, feat)
  s = tf.shape(x)
  s2 = x.get_shape().as_list()
  x = tf.expand_dims(x, 2)
  x = tf.expand_dims(x, 4)
  x = tf.tile(x, [1, 1, factor, 1, factor, 1])
  x = tf.reshape(x, [s[0], factor * s[1], factor * s[2], s[3]])
  if s2[1] is not None:
    s2[1] *= factor
  if s2[2] is not None:
    s2[2] *= factor
  x.set_shape(s2)
  return x


def bootstrapped_ce_loss(ce, fraction):
  # only consider k worst pixels (lowest posterior probability) per image
  assert fraction is not None
  batch_size = ce.get_shape().as_list()[0]
  if batch_size is None:
    batch_size = tf.shape(ce)[0]
  k = tf.maximum(tf.cast(tf.cast(tf.shape(ce)[1] * tf.shape(ce)[2], tf.float32) * fraction, tf.int32), 1)
  bs_ce, _ = tf.nn.top_k(tf.reshape(ce, shape=[batch_size, -1]), k=k, sorted=False)
  bs_ce = tf.reduce_mean(bs_ce, axis=1)
  bs_ce = tf.reduce_sum(bs_ce, axis=0)
  return bs_ce


def bootstrapped_ce_loss_ignore_void_label(ce, fraction, no_void_label_mask):
  # only consider k worst pixels (lowest posterior probability) per image
  assert fraction is not None
  batch_size = ce.get_shape().as_list()[0]
  if batch_size is None:
    batch_size = tf.shape(ce)[0]
  no_void_label_count = tf.cast(tf.cast(tf.reduce_sum(tf.cast(tf.reshape(no_void_label_mask, [batch_size,-1]),
                                                              tf.int32), axis=1), tf.float32) * fraction, tf.int32)
  # no_void_label_count = tf.Print(no_void_label_count, [no_void_label_count])
  result = tf.map_fn(lambda x: tf.reduce_mean(tf.nn.top_k(tf.reshape(ce[x], shape=[-1]),
                                           k=tf.maximum(tf.reshape(no_void_label_count[x],shape=[]), 1),
                                           sorted=False)[0], axis=0), tf.range(0, batch_size), dtype=tf.float32)

  # bs_ce = tf.reduce_mean(result, axis=1)
  bs_ce = tf.reduce_sum(result, axis=0)
  return bs_ce


def weighted_clicks_loss(y_pred, targets, x_image, loss_str, add_summary_fn):
  assert x_image.get_shape().as_list()[-1] == 8
  sigma = 1.0
  dist_to_use = "gaussian"
  weightage_copy = 1
  weightage_current = 2
  monotonicity = False
  locality = True
  apply_weights = True

  if "sigma" in loss_str:
    sigma = float(loss_str.split("sigma-")[1].split(":")[0])
  if "dist" in loss_str:
    dist_to_use = loss_str.split("dist-")[1].split(":")[0]
  if "weightage_current" in loss_str:
    weightage_current = float(loss_str.split("weightage_current-")[1].split(":")[0])
  if "weightage_copy" in loss_str:
    weightage_copy = float(loss_str.split("weightage_copy-")[1].split(":")[0])
  if "monotonicity" in loss_str:
    monotonicity = True if loss_str.split("monotonicity-")[1].split(":")[0] == "True" else False
  if "locality" in loss_str:
    locality = False if loss_str.split("locality-")[1].split(":")[0] == "False" else True
  if "apply_weights" in loss_str:
    apply_weights = False if loss_str.split("apply_weights-")[1].split(":")[0] == "False" else True
    
  if dist_to_use == "laplace":
    dist = tf.contrib.distributions.Laplace(loc=0., scale=sigma)
  else:
    dist = tf.contrib.distributions.Normal(loc=0., scale=sigma)

  # Copy old label in the context click region
  old_label = tf.expand_dims(x_image[:, :, :, 3], axis=3)
  
  target_old_label = tf.cast(tf.image.resize_bilinear(old_label, tf.shape(y_pred)[1:3]), tf.int32)[:, :, :, 0]
  ce_copy_label = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                 labels=target_old_label, name="ce_copy_label")
  u0_1 = tf.expand_dims(x_image[:, :, :, 5], axis=3)
  u1_1 = tf.expand_dims(x_image[:, :, :, 7], axis=3)

  mask_copy_label = tf.expand_dims(tf.ones_like(ce_copy_label), axis=3)
  if locality:
    mask_copy_label = get_locality_map(u0_1, u1_1, ce_copy_label)

  #ce_copy_label *= mask_copy_label[:, :, :, 0]
  add_summary_fn(tf.expand_dims(ce_copy_label, axis=3), "ce_copy_label")

  context_clicks = tf.concat([u0_1,
                              u1_1], axis=3)
  weights_copy_label = tf.reduce_max(tf.image.resize_bilinear(dist.prob(context_clicks),
                                                              tf.shape(ce_copy_label)[1:3]),
                                     axis=-1)

  # calculate loss for the current click
  ce_current_click = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                    labels=targets, name="ce_current_click")
  u0_0 = tf.expand_dims(x_image[:, :, :, 4], axis=3)
  u1_0 = tf.expand_dims(x_image[:, :, :, 6], axis=3)
  # condition = tf.logical_and(tf.greater(u0_0, locality_threshold),
  #                            tf.greater(u1_0, locality_threshold))
  # # ignore regions outside the threshold
  # mask_current_click = tf.image.resize_bilinear(tf.where(condition, tf.zeros_like(u0_0), tf.ones_like(u0_0)),
  #                                               tf.shape(ce_current_click)[1:3])
  mask_current_click = tf.expand_dims(tf.ones_like(ce_current_click), axis=3)
  if locality:
    mask_current_click = get_locality_map(u0_0, u1_0, ce_current_click)
  if monotonicity:
    monotonicity_map = get_monotonicity_map(u0_0, u1_0, y_pred, x_image)
    mask_current_click *= monotonicity_map
    #reset copy label mask and corresponding weights to get strong copy label influence.
    mask_copy_label = tf.where(tf.equal(monotonicity_map, 0), tf.ones_like(monotonicity_map), mask_copy_label)
    weights_copy_label = tf.where(tf.equal(monotonicity_map[:, :, :, 0], 0), tf.ones_like(monotonicity_map[:, :, :,0], dtype=tf.float32)*dist.prob(0.0), weights_copy_label)
  
  ce_copy_label *= mask_copy_label[:, :, :, 0]
  ce_current_click *= mask_current_click[:, :, :, 0]
  add_summary_fn(tf.expand_dims(ce_current_click, axis=3), "ce_current_click")

  current_click = tf.concat([u0_0,
                             u1_0], axis=3)
  weights_current_click = tf.reduce_max(tf.image.resize_bilinear(dist.prob(current_click),
                                                                 tf.shape(ce_current_click)[1:3]),
                                        axis = -1)
  normalisation_factor = tf.reduce_sum(weights_copy_label + weights_current_click, axis=[1,2])
  # normalisation_factor = tf.reduce_sum(weights_current_click, axis=[1,2])

  weights_current_click = tf.map_fn(lambda x: weights_current_click[x] / tf.maximum(tf.cast(normalisation_factor[x],
                                                                                            tf.float32), 1e-10),
                                    tf.range(0, tf.shape(normalisation_factor)[0]), dtype=tf.float32)
  weights_copy_label = tf.map_fn(lambda x: weights_copy_label[x] / tf.maximum(tf.cast(normalisation_factor[x],
                                                                                      tf.float32), 1e-10),
                                 tf.range(0, tf.shape(normalisation_factor)[0]), dtype=tf.float32)

  add_summary_fn(tf.expand_dims(weights_copy_label, axis=3), "weights_copy_label")
  add_summary_fn(tf.expand_dims(weights_current_click, axis=3), "weights_current_click")
  add_summary_fn(tf.cast(tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=3), tf.float32), "predicted label")
 
  if apply_weights: 
    ce_copy_label *= weights_copy_label
    ce_current_click *= weights_current_click

  ce_loss_map = ((weightage_current * ce_current_click) + (weightage_copy * ce_copy_label)) / \
                (weightage_current + weightage_copy)

  # Apply monotonicity to the whole loss map.
  #if monotonicity:
  #  ce_loss_map *= get_monotonicity_map(u0_0, u1_0, y_pred, x_image)[:, :, :, 0]

  add_summary_fn(tf.expand_dims(ce_loss_map, axis=3), "loss_map")
  if apply_weights:
    weighted_ce = tf.reduce_sum(ce_loss_map, axis=[1, 2])
  else:
    weighted_ce = tf.reduce_mean(ce_loss_map, axis=[1, 2])

  ce = tf.reduce_sum(weighted_ce, axis=0)

  return ce, tf.expand_dims(ce_loss_map, axis=3)


def get_monotonicity_map(u0, u1, y_pred, x_image):
  outputs = tf.expand_dims(tf.argmax(y_pred, axis = -1), axis=3)
  result_map = tf.ones_like(outputs)
  old_label = tf.image.resize_bilinear(tf.expand_dims(x_image[:, :, :, 3], axis=3),
                                       tf.shape(outputs)[1:3]) 
  i = tf.cast(tf.logical_and(tf.cast(outputs, tf.bool), tf.cast(old_label, tf.bool)), tf.float32)
  u = tf.cast(tf.logical_or(tf.cast(outputs, tf.bool), tf.cast(old_label, tf.bool)), tf.float32)

  result_map = tf.map_fn(lambda x: tf.cond(tf.equal(tf.reduce_min(u0[x]), 0), lambda: tf.subtract(u[x], old_label[x]),
                                           lambda: tf.subtract(old_label[x], i[x])),
                         tf.range(0, tf.shape(outputs)[0]), tf.float32)
  result_map = tf.where(tf.not_equal(result_map, 0), tf.zeros_like(result_map), tf.ones_like(result_map))
    
  result_map = tf.cast(result_map, tf.float32)

  return result_map


def binary_balanced_bootstrapped_ce_loss(logits, targets, fraction):
  #in contrast to bootstrapped_ce_loss, we take the logits as input and don't support batching
  #make tensors flat
  logits = tf.reshape(logits, [-1])
  targets = tf.reshape(targets, [-1])
  targets = tf.cast(targets, tf.float32)
  targets_inv = 1.0 - targets

  k_pos = tf.maximum(tf.cast(tf.reduce_sum(targets) * fraction, tf.int32), 1)
  k_neg = tf.maximum(tf.cast(tf.reduce_sum(targets_inv) * fraction, tf.int32), 1)
  ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
  ce_pos = ce * targets
  ce_neg = ce * targets_inv

  bs_ce_pos, _ = tf.nn.top_k(ce_pos, k=k_pos, sorted=False)
  bs_ce_neg, _ = tf.nn.top_k(ce_neg, k=k_neg, sorted=False)
  bs_ce_pos = tf.reduce_mean(bs_ce_pos)
  bs_ce_neg = tf.reduce_mean(bs_ce_neg)
  bs_ce = bs_ce_pos + bs_ce_neg
  return bs_ce
