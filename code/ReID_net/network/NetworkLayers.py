import numpy
import tensorflow as tf
from tensorflow.python.training import moving_averages
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

from .Util_Network import conv2d, max_pool, global_avg_pool, apply_dropout, prepare_input, \
  prepare_collapsed_input_and_dropout, get_activation, create_batch_norm_vars, create_bilinear_upsampling_weights, \
  conv2d_dilated

BATCH_NORM_DECAY_DEFAULT = 0.95
BATCH_NORM_EPSILON = 1e-5
L2_DEFAULT = 1e-4


class Layer(object):
  output_layer = False

  def __init__(self):
    self.summaries = []
    self.regularizers = []
    self.update_ops = []
    self.n_params = 0

  def add_scalar_summary(self, op, name):
    summary = tf.summary.scalar(name, op)
    self.summaries.append(summary)

  def add_image_summary(self, im, name):
    summary = tf.summary.image(name, im)
    self.summaries.append(summary)

  def add_mask_summary(self, mask, name):
    from ReID_net.datasets.Util.Util import smart_shape
    assert len(smart_shape(mask)) == 2
    im = tf.tile(tf.cast(mask, tf.float32)[tf.newaxis, :, :, tf.newaxis], multiples=[1, 1, 1, 3])
    summary = tf.summary.image(name, im)
    self.summaries.append(summary)

  def create_and_apply_batch_norm(self, inp, n_features, decay, tower_setup, scope_name="bn"):
    beta, gamma, moving_mean, moving_var = create_batch_norm_vars(n_features, tower_setup, scope_name)
    self.n_params += 2 * n_features
    if tower_setup.is_main_train_tower:
      assert tower_setup.is_training
    if tower_setup.is_training and not tower_setup.freeze_batchnorm:
      xn, batch_mean, batch_var = tf.nn.fused_batch_norm(inp, gamma, beta, epsilon=BATCH_NORM_EPSILON, is_training=True)
      if tower_setup.is_main_train_tower:
        update_op1 = moving_averages.assign_moving_average(
          moving_mean, batch_mean, decay, zero_debias=False, name='mean_ema_op')
        update_op2 = moving_averages.assign_moving_average(
          moving_var, batch_var, decay, zero_debias=False, name='var_ema_op')
        if tower_setup.use_update_ops_collection:
          tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
          tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        else:
          self.update_ops.append(update_op1)
          self.update_ops.append(update_op2)
      return xn
    else:
      xn = tf.nn.batch_normalization(inp, moving_mean, moving_var, beta, gamma, BATCH_NORM_EPSILON)
      return xn

  def create_weight_variable(self, name, shape, l2, tower_setup, trainable=True, initializer=None):
    with tf.device(tower_setup.variable_device):
      if initializer is None:
        # He initialization
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
      self.n_params += numpy.prod(shape)
      W = tf.get_variable(name, shape, tower_setup.dtype, initializer, trainable=trainable)
      if l2 > 0.0:
        self.regularizers.append(l2 * tf.nn.l2_loss(W))
      if tower_setup.use_weight_summaries:
        summ = tf.summary.histogram(name, W)
        self.summaries.append(summ)
        self.add_scalar_summary(tf.reduce_max(tf.abs(W)), name + "/W_abs_max")
      return W

  # adapted from https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
  def create_transposed_conv_weight_variable(self, name, shape, l2, tower_setup, trainable=True):
    with tf.device(tower_setup.variable_device):
      weights = create_bilinear_upsampling_weights(shape)
      initializer = tf.constant_initializer(value=weights, dtype=tf.float32)
      self.n_params += numpy.prod(shape)
      W = tf.get_variable(name, shape, tower_setup.dtype, initializer,trainable=trainable)
      if l2 > 0.0:
        self.regularizers.append(l2 * tf.nn.l2_loss(W))
      if tower_setup.use_weight_summaries:
        summ = tf.summary.histogram(name, W)
        self.summaries.append(summ)
        self.add_scalar_summary(tf.reduce_max(tf.abs(W)), name + "/W_abs_max")
      return W

  def create_bias_variable(self, name, shape, tower_setup, trainable=True, initializer=None):
    with tf.device(tower_setup.variable_device):
      if initializer is None:
        initializer = tf.constant_initializer(0.0, dtype=tower_setup.dtype)
      self.n_params += numpy.prod(shape)
      b = tf.get_variable(name, shape, tower_setup.dtype, initializer, trainable=trainable)
      if tower_setup.use_weight_summaries:
        summ = tf.summary.histogram(name, b)
        self.summaries.append(summ)
        self.add_scalar_summary(tf.reduce_max(tf.abs(b)), name + "/b_abs_max")
      return b


class Conv(Layer):
  def __init__(self, name, inputs, n_features, tower_setup, old_order=False, filter_size=(3, 3),
               strides=(1, 1), dilation=None, pool_size=(1, 1), pool_strides=None, activation="relu", dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=BATCH_NORM_DECAY_DEFAULT, l2=L2_DEFAULT):
    super(Conv, self).__init__()
    # mind the order of dropout, conv, activation and batchnorm!
    # default: batchnorm -> activation -> dropout -> conv -> pool
    # if old_order: dropout -> conv -> batchnorm -> activation -> pool

    curr, n_features_inp = prepare_input(inputs)

    filter_size = list(filter_size)
    strides = list(strides)
    pool_size = list(pool_size)
    if pool_strides is None:
      pool_strides = pool_size

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup)
      b = None
      if bias:
        b = self.create_bias_variable("b", [n_features], tower_setup)

      if old_order:
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides)
        else:
          curr = conv2d_dilated(curr, W, dilation)
        if bias:
          curr += b
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
      else:
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides)
        else:
          curr = conv2d_dilated(curr, W, dilation)
        if bias:
          curr += b

      if pool_size != [1, 1]:
        curr = max_pool(curr, pool_size, pool_strides)
    self.outputs = [curr]


class ResidualUnit2(Layer):
  def __init__(self, name, inputs, tower_setup, n_convs=2, n_features=None, dilations=None, strides=None,
               filter_size=None, activation="relu", dropout=0.0, batch_norm_decay=BATCH_NORM_DECAY_DEFAULT,
               l2=L2_DEFAULT):
    super(ResidualUnit2, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    res = curr
    assert n_convs >= 1, n_convs

    if dilations is not None:
      assert strides is None
    elif strides is None:
      strides = [[1, 1]] * n_convs
    if filter_size is None:
      filter_size = [[3, 3]] * n_convs
    if n_features is None:
      n_features = n_features_inp
    if not isinstance(n_features, list):
      n_features = [n_features] * n_convs

    with tf.variable_scope(name):
      curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup, "bn0")
      curr = get_activation(activation)(curr)
      curr = apply_dropout(curr, dropout)

      if strides is None:
        strides_res = [1, 1]
      else:
        strides_res = numpy.prod(strides, axis=0).tolist()
      if (n_features[-1] != n_features_inp) or (strides_res != [1, 1]):
        W0 = self.create_weight_variable("W0", [1, 1] + [n_features_inp, n_features[-1]], l2, tower_setup)
        if dilations is None:
          res = conv2d(curr, W0, strides_res)
        else:
          res = conv2d(curr, W0)

      W1 = self.create_weight_variable("W1", filter_size[0] + [n_features_inp, n_features[0]], l2, tower_setup)
      if dilations is None:
        curr = conv2d(curr, W1, strides[0])
      else:
        curr = conv2d_dilated(curr, W1, dilations[0])
      for idx in range(1, n_convs):
        curr = self.create_and_apply_batch_norm(curr, n_features[idx - 1], batch_norm_decay,
                                                tower_setup, "bn" + str(idx + 1))
        curr = get_activation(activation)(curr)
        Wi = self.create_weight_variable("W" + str(idx + 1), filter_size[idx] + [n_features[idx - 1], n_features[idx]],
                                         l2, tower_setup)
        if dilations is None:
          curr = conv2d(curr, Wi, strides[idx])
        else:
          curr = conv2d_dilated(curr, Wi, dilations[idx])

    curr += res
    self.outputs = [curr]


class Upsampling(Layer):
  def __init__(self, name, inputs, tower_setup, n_features, concat, activation="relu",
               filter_size=(3, 3), batch_norm_decay=BATCH_NORM_DECAY_DEFAULT, l2=L2_DEFAULT):
    super(Upsampling, self).__init__()
    filter_size = list(filter_size)
    assert isinstance(concat, list)
    assert len(concat) > 0
    curr, n_features_inp = prepare_input(inputs)
    concat_inp, n_features_concat = prepare_input(concat)

    curr = tf.image.resize_nearest_neighbor(curr, tf.shape(concat_inp)[1:3])
    curr = tf.concat([curr, concat_inp], axis=3)
    n_features_curr = n_features_inp + n_features_concat

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_curr, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      curr = conv2d(curr, W) + b
      curr = get_activation(activation)(curr)

    self.outputs = [curr]


class FullyConnected(Layer):
  def __init__(self, name, inputs, n_features, tower_setup, activation="relu", dropout=0.0, batch_norm=False,
               batch_norm_decay=BATCH_NORM_DECAY_DEFAULT, l2=L2_DEFAULT):
    super(FullyConnected, self).__init__()
    inp, n_features_inp = prepare_collapsed_input_and_dropout(inputs, dropout)
    with tf.variable_scope(name):
      if batch_norm:
        inp = tf.expand_dims(inp, axis=0)
        inp = tf.expand_dims(inp, axis=0)
        inp = self.create_and_apply_batch_norm(inp, n_features_inp, batch_norm_decay, tower_setup)
        inp = tf.squeeze(inp, axis=[0, 1])
      W = self.create_weight_variable("W", [n_features_inp, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      z = tf.matmul(inp, W) + b
      h = get_activation(activation)(z)
    self.outputs = [h]
    self.n_features = n_features


class Collapse(Layer):
  def __init__(self, name, inputs, tower_setup, activation="relu", batch_norm_decay=BATCH_NORM_DECAY_DEFAULT):
    super(Collapse, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    with tf.variable_scope(name):
      inp = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup)
      h_act = get_activation(activation)(inp)
      out = global_avg_pool(h_act)
    self.outputs = [out]


#not really a layer, but we can handle it the same way
class ResNet50(Layer):
  def __init__(self, name, inputs, tower_setup, for_imagenet_classification=False):
    super(ResNet50, self).__init__()
    #for now always freeze the batch norm of the resnet
    inp, n_features_inp = prepare_input(inputs)
    #for grayscale
    if n_features_inp == 1:
      inp = tf.concat([inp, inp, inp], axis=-1)
    else:
      assert n_features_inp == 3
    #to keep the preprocessing consistent with our usual preprocessing, revert the std normalization
    from ReID_net.datasets.Util.Normalization import IMAGENET_RGB_STD
    #I double checked it, this seems to be the right preprocessing
    inp = inp * IMAGENET_RGB_STD * 255

    num_classes = 1000 if for_imagenet_classification else None
    #note that we do not add the name to the variable scope at the moment, so that if we would use multiple resnets
    #in the same network, this will throw an error.
    #but if we add the name, the loading of pretrained weights will be difficult
    with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
      with slim.arg_scope([slim.model_variable, slim.variable], device=tower_setup.variable_device):
        logits, end_points = slim.nets.resnet_v1.resnet_v1_50(inp, num_classes=num_classes, is_training=False)
    #mapping from https://github.com/wuzheng-sjtu/FastFPN/blob/master/libs/nets/pyramid_network.py
    mapping = {"C1": "resnet_v1_50/conv1/Relu:0",
               "C2": "resnet_v1_50/block1/unit_2/bottleneck_v1",
               "C3": "resnet_v1_50/block2/unit_3/bottleneck_v1",
               "C4": "resnet_v1_50/block3/unit_5/bottleneck_v1",
               "C5": "resnet_v1_50/block4/unit_3/bottleneck_v1"}
    if for_imagenet_classification:
      self.outputs = [tf.nn.softmax(logits)]
    else:
      # use C3 up to C5
      self.outputs = [end_points[mapping[c]] for c in ["C3", "C4", "C5"]]
    self.n_params = 25600000  # roughly 25.6M


class ResNet50WithImageNetOutput(ResNet50):
  #this layer is not meant for training, but just to test out the initialization
  output_layer = True

  def __init__(self, name, inputs, tower_setup, targets, n_classes):
    super(ResNet50WithImageNetOutput, self).__init__(name, inputs, tower_setup, for_imagenet_classification=True)
    self.loss = tf.constant(0, dtype=tf.float32)
    self.measures = {}
