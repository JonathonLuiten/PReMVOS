import numpy
import tensorflow as tf

import ReID_net.Constants as Constants
from ReID_net.Measures import create_confusion_matrix, get_average_precision, compute_binary_ious_tf
from .NetworkLayers import Layer, L2_DEFAULT, BATCH_NORM_DECAY_DEFAULT
from .Util_Network import prepare_input, global_avg_pool, prepare_collapsed_input_and_dropout, get_activation, \
  apply_dropout, conv2d, conv2d_dilated, bootstrapped_ce_loss, bootstrapped_ce_loss_ignore_void_label, \
  weighted_clicks_loss
from ReID_net.datasets.Util.Util import smart_shape

MAX_ADJUSTABLE_CLASSES = 100  # max 100 objects per sequence should be sufficient


class Softmax(Layer):
  output_layer = True

  def __init__(self, name, inputs, targets, n_classes, tower_setup, global_average_pooling=False, dropout=0.0,
               loss="ce", l2=L2_DEFAULT):
    super(Softmax, self).__init__()
    self.measures = {}
    if global_average_pooling:
      inp, n_features_inp = prepare_input(inputs)
      inp = global_avg_pool(inp)
    else:
      inp, n_features_inp = prepare_collapsed_input_and_dropout(inputs, dropout)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", [n_features_inp, n_classes], l2, tower_setup)
      b = self.create_bias_variable("b", [n_classes], tower_setup)
      y_ref = tf.cast(targets, tf.int64)
      y_pred = tf.matmul(inp, W) + b
      self.outputs = [tf.nn.softmax(y_pred, -1, 'softmax')]
      errors = tf.not_equal(tf.argmax(y_pred, 1), y_ref)
      errors = tf.reduce_sum(tf.cast(errors, tower_setup.dtype))
      self.measures['errors'] = errors

      if loss == "ce":
        cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=y_pred, labels=y_ref, name='cross_entropy_per_example')
        self.loss = tf.reduce_sum(cross_entropy_per_example, name='cross_entropy_sum')
      else:
        assert False, "Unknown loss " + loss

      self.add_scalar_summary(self.loss, "loss")


class SimilaritySoftmax(Layer):
  output_layer = False

  def __init__(self, name, inputs, n_classes, tower_setup, dropout=0.0, l2=L2_DEFAULT):
    super(SimilaritySoftmax, self).__init__()


    inp, n_features_inp = prepare_collapsed_input_and_dropout(inputs, dropout)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", [n_features_inp, n_classes], l2, tower_setup)
      b = self.create_bias_variable("b", [n_classes], tower_setup)

      self.y_pred = tf.matmul(inp, W) + b
      output = tf.nn.softmax(self.y_pred, -1, 'softmax')
      self.outputs = [output]
      self.out_labels = tf.argmax(output,1)


class SimilaritySoftmaxOutput(SimilaritySoftmax):
  output_layer = True

  def __init__(self, name, inputs, targets, n_classes, tower_setup, dropout=0.0,
               loss="ce", targets_are_duplicated=False, expandedV1=False, l2=L2_DEFAULT):
    super(SimilaritySoftmaxOutput, self).__init__(name, inputs, n_classes, tower_setup, dropout, l2)
    if targets_are_duplicated:
      targets = targets[::2]
    self.measures = {}

    y_ref = tf.cast(targets, tf.int64)
    errors = tf.not_equal(tf.argmax(self.y_pred, 1), y_ref)
    errors = tf.reduce_sum(tf.cast(errors, tower_setup.dtype))
    if targets_are_duplicated:
      errors *= 2
    elif expandedV1:
      # TODO: Put in proper factor: (groupsize - 1)*(batchsize/groupsize - 1)
      # factor = 14 ## For ExpandedSiameseMerge (groupsize - 1)*(batchsize/groupsize - 1)
      factor = 64  ## For CompleteSiameseMerge (batchsize, if include diag) (batchsize-1, if not)
      errors /= factor
    self.measures['errors'] = errors

    if loss == "ce":
      cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.y_pred, labels=y_ref, name='cross_entropy_per_example')
      self.loss = tf.reduce_sum(cross_entropy_per_example, name='cross_entropy_sum')
      if targets_are_duplicated:
        self.loss *= 2
      elif expandedV1:
        self.loss /= factor
    else:
      assert False, "Unknown loss " + loss

    self.add_scalar_summary(self.loss, "loss")


class SegmentationSoftmax(Layer):
  output_layer = True

  def create_weights(self, n_classes, filter_size, n_features_inp, l2, tower_setup):
    if n_classes is None:
      n_class_weights = 2
    else:
      n_class_weights = n_classes
    W = self.create_weight_variable("W", filter_size + [n_features_inp, n_class_weights], l2, tower_setup)
    b = self.create_bias_variable("b", [n_class_weights], tower_setup)

    W_used = W
    b_used = b
    if n_classes is None:
      with tf.device(tower_setup.variable_device):
        n_classes_current = tf.get_variable("n_classes_current", shape=[], trainable=False, dtype=tf.int32)
        W_adjustable = tf.get_variable("W_adjustable", filter_size + [n_features_inp, MAX_ADJUSTABLE_CLASSES])
        b_adjustable = tf.get_variable("b_adjustable", [MAX_ADJUSTABLE_CLASSES])
        if l2 > 0.0:
          self.regularizers.append(l2 * tf.nn.l2_loss(W_adjustable))
        W_used = W_adjustable[..., :n_classes_current]
        b_used = b_adjustable[:n_classes_current]
    else:
      W_adjustable = b_adjustable = n_classes_current = None
    return W, b, W_adjustable, b_adjustable, n_classes_current, W_used, b_used

  def _create_adjustable_output_assign_data(self, tower_setup):
    if self.W_adjustable is None or self.b_adjustable is None:
      return None
    else:
      W_adjustable_val_placeholder = tf.placeholder(tower_setup.dtype, name="W_adjustable_val_placeholder")
      b_adjustable_val_placeholder = tf.placeholder(tower_setup.dtype, name="b_adjustable_val_placeholder")
      n_classes_current_val_placeholder = tf.placeholder(tf.int32, name="n_classes_current_val_placeholder")
      assign_W_adjustable = tf.assign(self.W_adjustable, W_adjustable_val_placeholder)
      assign_b_adjustable = tf.assign(self.b_adjustable, b_adjustable_val_placeholder)
      assign_n_classes_current = tf.assign(self.n_classes_current, n_classes_current_val_placeholder)
      return assign_W_adjustable, assign_b_adjustable, assign_n_classes_current, W_adjustable_val_placeholder, \
          b_adjustable_val_placeholder, n_classes_current_val_placeholder

  def adjust_weights_for_multiple_objects(self, session, n_objects):
    W_val, b_val = session.run([self.W, self.b])
    W_adjustable_val_new = numpy.zeros(W_val.shape[:-1] + (MAX_ADJUSTABLE_CLASSES,), dtype="float32")
    b_adjustable_val_new = numpy.zeros(MAX_ADJUSTABLE_CLASSES, dtype="float32")

    W_adjustable_val_new[..., :n_objects + 1] = W_val[..., [0] + ([1] * n_objects)]
    b_adjustable_val_new[:n_objects + 1] = b_val[[0] + ([1] * n_objects)]
    b_adjustable_val_new[1:n_objects + 1] -= numpy.log(n_objects)

    assign_W_adjustable, assign_b_adjustable, assign_n_classes_current, W_adjustable_val_placeholder, \
        b_adjustable_val_placeholder, n_classes_current_val_placeholder = self.adjustable_output_assign_data

    session.run([assign_W_adjustable, assign_b_adjustable, assign_n_classes_current],
                feed_dict={W_adjustable_val_placeholder: W_adjustable_val_new,
                           b_adjustable_val_placeholder: b_adjustable_val_new,
                           n_classes_current_val_placeholder: n_objects + 1})

  def create_loss(self, loss_str, fraction, no_void_label_mask, targets, tower_setup, void_label, y_pred, x_image):
    ce = None
    if "ce" in loss_str:
      ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=targets, name="ce")

      if void_label is not None:
        mask = tf.cast(no_void_label_mask, tower_setup.dtype)
        ce *= mask
    if loss_str == "ce":
      ce = tf.reduce_mean(ce, axis=[1, 2])
      ce = tf.reduce_sum(ce, axis=0)
      loss = ce
    elif loss_str == "bootstrapped_ce":
      bs_ce = bootstrapped_ce_loss(ce, fraction)
      loss = bs_ce
    # take fraction with respect to the region without void label
    elif loss_str == "bootstrapped_ignore_void_label_ce":
      bs_ce = bootstrapped_ce_loss_ignore_void_label(ce, fraction, no_void_label_mask)
      loss = bs_ce
    elif "weighted_clicks_loss" in loss_str:
      bs_ce, loss_map = weighted_clicks_loss(y_pred, targets, x_image, loss_str, self.add_image_summary)
      loss = bs_ce
    else:
      assert False, "Unknown loss " + loss_str
    return loss

  def create_measures(self, n_classes, pred, targets):
    measures = {}
    conf_matrix = tf.py_func(create_confusion_matrix, [pred, targets, self.n_classes_current], [tf.int64])
    measures[Constants.CONFUSION_MATRIX] = conf_matrix[0]
    # Calculate mAP for binary segmentation.
    if n_classes == 2:
      ap = tf.py_func(get_average_precision, [targets, self.outputs, conf_matrix],
                      [tf.float64])
      measures[Constants.AP] = ap

      binary_iou = tf.py_func(compute_binary_ious_tf, [targets, pred], [tf.float32])
      measures[Constants.BINARY_IOU] = binary_iou
    return measures

  def __init__(self, name, inputs, targets, n_classes, void_label, tower_setup, filter_size=(1, 1),
               input_activation=None, dilation=None, resize_targets=False, resize_logits=False, loss="ce",
               fraction=None, l2=L2_DEFAULT, dropout=0.0, imgs_raw = None):
    super(SegmentationSoftmax, self).__init__()
    assert targets.get_shape().ndims == 4, targets.get_shape()
    assert not (resize_targets and resize_logits)
    inp, n_features_inp = prepare_input(inputs)

    filter_size = list(filter_size)

    with tf.variable_scope(name):
      if input_activation is not None:
        inp = get_activation(input_activation)(inp)

      inp = apply_dropout(inp, dropout)

      self.W, self.b, self.W_adjustable, self.b_adjustable, self.n_classes_current, W, b = self.create_weights(
        n_classes, filter_size, n_features_inp, l2, tower_setup)
      self.adjustable_output_assign_data = self._create_adjustable_output_assign_data(tower_setup)
      if self.n_classes_current is None:
        self.n_classes_current = n_classes

      if dilation is None:
        y_pred = conv2d(inp, W) + b
      else:
        y_pred = conv2d_dilated(inp, W, dilation) + b
      self.outputs = [tf.nn.softmax(y_pred, -1, 'softmax')]

      if resize_targets:
        targets = tf.image.resize_nearest_neighbor(targets, tf.shape(y_pred)[1:3])
      if resize_logits:
        y_pred = tf.image.resize_images(y_pred, tf.shape(targets)[1:3])

      pred = tf.argmax(y_pred, axis=3)
      targets = tf.cast(targets, tf.int64)
      targets = tf.squeeze(targets, axis=3)

      # TODO: Void label is not considered in the iou calculation.
      if void_label is not None:
        # avoid nan by replacing void label by 0
        # note: the loss for these cases is multiplied by 0 below
        void_label_mask = tf.equal(targets, void_label)
        no_void_label_mask = tf.logical_not(void_label_mask)
        targets = tf.where(void_label_mask, tf.zeros_like(targets), targets)
      else:
        no_void_label_mask = None

      self.measures = self.create_measures(n_classes, pred, targets)
      self.loss = self.create_loss(loss, fraction, no_void_label_mask, targets,
                                   tower_setup, void_label, y_pred, imgs_raw)
      self.add_image_summary(tf.cast(tf.expand_dims(pred, axis=3), tf.float32), "predicted label")
      self.add_scalar_summary(self.loss, "loss")


class FullyConnectedWithTripletLoss(Layer):
  output_layer = True

  def __init__(self, name, inputs, targets, n_classes, n_features, tower_setup, imgs_raw = None, original_labels=None,
               activation="linear", dropout=0.0, batch_norm=False, batch_norm_decay=BATCH_NORM_DECAY_DEFAULT,
               l2=L2_DEFAULT, negative_weighting_factor=1):
    super(FullyConnectedWithTripletLoss, self).__init__()
    self.measures = {}
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

      if original_labels is not None:
        self.measures[Constants.EMBEDDING] = [h]
        self.measures[Constants.ORIGINAL_LABELS] = [original_labels]

      self.add_scalar_summary(tf.norm(h[0]), "embedding_norm")
      self.summaries.append(tf.summary.histogram("embedding", h))

      size = smart_shape(h)[0]
      eps = 1e-10

      # New print debug example
      def my_print(x, name):
        with tf.control_dependencies([tf.assert_equal(tf.reduce_all(tf.greater(tf.shape(x), 0)), True)]):
          if x.dtype in (tf.float32, tf.float64):
            with tf.control_dependencies([tf.assert_equal(tf.reduce_all(tf.is_finite(x)), True)]):
              return tf.Print(x, [tf.shape(x), tf.reduce_all(tf.is_finite(x)), x], name, summarize=200)
          else:
            return tf.Print(x, [tf.shape(x), x], name)

      def get_loss(idx):
        anchor = h[idx, :]
        anchor_class = targets[idx]

        ###### New code ######
        class_division = tf.equal(targets, anchor_class)
        not_self_mask = tf.logical_not(tf.cast(tf.one_hot(idx, depth=size),tf.bool))
        positive_output = tf.boolean_mask(h, tf.logical_and(class_division, not_self_mask))
        negative_output = tf.boolean_mask(h, tf.logical_not(class_division))
        # negative_output = tf.boolean_mask(h, tf.logical_and(tf.logical_not(class_division),not_self_mask))
        # positive_output = my_print(positive_output,"positive_output")
        # negative_output = my_print(negative_output, "negative_output")

        positive_distances = tf.abs(anchor - positive_output)
        pos_dis_val = tf.norm(positive_distances + eps,axis=1)
        hardest_positive, hardest_positive_idx = tf.nn.top_k(pos_dis_val,1)

        negative_distances = tf.abs(anchor - negative_output)
        neg_dis_val = tf.norm(negative_distances + eps, axis=1)
        minus_neg_dis_val = tf.negative(neg_dis_val)
        # minus_neg_dis_val = tf.Print(minus_neg_dis_val,[minus_neg_dis_val])
        # minus_neg_dis_val = tf.Print(minus_neg_dis_val, [minus_neg_dis_val.shape])
        minus_hardest_negative, hardest_negative_idx = tf.nn.top_k(minus_neg_dis_val, 1)
        hardest_negative = tf.negative(minus_hardest_negative)

        # minus_hardest_negative, hardest_negative_idx = tf.nn.top_k(minus_neg_dis_val, negative_weighting_factor)
        # hardest_negative = tf.negative(minus_hardest_negative)
        # hardest_negative = tf.reduce_sum(hardest_negative,-1)

        ###### Old code with dynamic partition ######
        # class_division = tf.cast(tf.equal(targets, anchor_class), tf.int32)
        # not_self_mask = tf.logical_not(tf.cast(tf.one_hot(idx, depth=size), tf.bool))
        # partitioned_output = tf.dynamic_partition(h, class_division, 2)
        # positive_output = partitioned_output[1]
        # negative_output = partitioned_output[0]

        # class_division = tf.equal(targets, anchor_class)
        # not_self_mask = tf.logical_not(tf.cast(tf.one_hot(idx, depth=size),tf.bool))
        # positive_output = tf.boolean_mask(h, tf.logical_and(class_division, not_self_mask))
        # negative_output = tf.boolean_mask(h, tf.logical_not(class_division))
        #
        #
        # positive_distances = tf.abs(anchor - positive_output)
        # pos_dis_val = tf.norm(positive_distances+eps, axis=1)
        # hardest_positive_idx = tf.argmax(pos_dis_val,0)
        # pos_div_size = smart_shape(positive_output)[0]
        # pos_divider = tf.one_hot(hardest_positive_idx,pos_div_size,dtype=tf.int32)
        # hardest_positive = tf.dynamic_partition(positive_distances,pos_divider,2)[1]
        # hardest_positive_class = tf.gather(targets, hardest_positive_idx)
        # hardest_positive = tf.norm(hardest_positive+eps, axis=1)
        #
        # negative_distances = tf.abs(anchor - negative_output)
        # neg_dis_val = tf.norm(negative_distances+eps, axis=1)
        # hardest_negative_idx = tf.argmin(neg_dis_val,0)
        # neg_div_size = smart_shape(negative_output)[0]
        # neg_divider = tf.one_hot(hardest_negative_idx,neg_div_size,dtype=tf.int32)
        # hardest_negative = tf.dynamic_partition(negative_distances,neg_divider,2)[1]
        # hardest_negative_class = tf.gather(targets,hardest_negative_idx)
        # hardest_negative = tf.norm(hardest_negative+eps, axis=1)

        # hardest_positive = my_print(hardest_positive,"hardest_positive")
        # hardest_negative = my_print(hardest_negative,"hardest_negative")

        #### Next two lines should be the same
        loss = tf.nn.softplus(hardest_positive - hardest_negative)
        # loss = tf.nn.softplus(hardest_positive - negative_weighting_factor*hardest_negative)
        # loss = tf.log1p(tf.exp(hardest_positive - hardest_negative))

        #### Code for using a hard margin rather than a softmargin
        # margin = 1
        # loss = tf.maximum(0., margin + hardest_positive - hardest_negative)

        anchor_img = tf.zeros([],tf.float32)
        hard_pos_img = tf.zeros([],tf.float32)
        hard_neg_img = tf.zeros([],tf.float32)
        if imgs_raw is not None:
          positive_images = tf.boolean_mask(imgs_raw, tf.logical_and(class_division, not_self_mask))
          negative_images = tf.boolean_mask(imgs_raw, tf.logical_not(class_division))
          anchor_img = imgs_raw[idx]
          hard_pos_img = positive_images[tf.squeeze(hardest_positive_idx)]
          hard_neg_img = negative_images[tf.squeeze(hardest_negative_idx)]

          # self.summaries.append(tf.summary.image("anchor_image", imgs_raw[idx]))
          # positive_images = tf.squeeze(tf.boolean_mask(imgs_raw, tf.logical_and(class_division, not_self_mask)))
          # negative_images = tf.squeeze(tf.boolean_mask(imgs_raw, tf.logical_not(class_division)))
          # self.summaries.append(tf.summary.image("hardest_postive_image",positive_images[hardest_positive_idx]))
          # self.summaries.append(tf.summary.image("hardest_negative_image", negative_images[hardest_negative_idx]))

        return loss, hardest_positive, hardest_negative, anchor_img, hard_pos_img, hard_neg_img

      #### Next two lines should be the same
      loss, hardest_positive, hardest_negative, anchor_imgs, hard_pos_imgs, hard_neg_imgs = \
        tf.map_fn(get_loss, tf.range(0, size), dtype=(tf.float32,tf.float32,tf.float32, tf.float32, tf.float32, tf.float32))
      # loss, hardest_positive, hardest_negative = [get_loss(idx) for idx in xrange(size)]

      self.loss = tf.reduce_sum(loss)
      hardest_positive = tf.reduce_sum(hardest_positive)
      hardest_negative = tf.reduce_sum(hardest_negative)
      self.add_scalar_summary(self.loss, "loss")
      self.add_scalar_summary(hardest_positive, "hardest_positive")
      self.add_scalar_summary(hardest_negative, "hardest_negative")
      # tf.summary.image()
      self.n_features = n_features

      if imgs_raw is not None:
        self.summaries.append(tf.summary.image("anchor_image", anchor_imgs))
        self.summaries.append(tf.summary.image("hardest_postive_image",hard_pos_imgs))
        self.summaries.append(tf.summary.image("hardest_negative_image", hard_neg_imgs))

      ## Print - debug example code
      # def test(a):
      #   print(a)
      #   return numpy.array([5], dtype="int32")
      #
      # t, = tf.py_func(test, [WHATEVER YOU WANT TO PRINT], [tf.int32])
      # with tf.control_dependencies([t]):
      #   loss = tf.identity(loss)

      ## New print debug example
      # def my_print(x, name):
      #   with tf.control_dependencies([tf.assert_equal(tf.reduce_all(tf.greater(tf.shape(x), 0)), True)]):
      #     if x.dtype in (tf.float32, tf.float64):
      #       with tf.control_dependencies([tf.assert_equal(tf.reduce_all(tf.is_finite(x)), True)]):
      #         return tf.Print(x, [tf.shape(x), tf.reduce_all(tf.is_finite(x)), x], name, summarize=200)
      #     else:
      #       return tf.Print(x, [tf.shape(x), x], name)


class SoftmaxNoLoss(Layer):
  output_layer = True

  def __init__(self, name, inputs, targets, n_classes, tower_setup, global_average_pooling=False, dropout=0.0,
               loss="ce", l2=L2_DEFAULT):
    super(SoftmaxNoLoss, self).__init__()
    self.measures = {}
    if global_average_pooling:
      inp, n_features_inp = prepare_input(inputs)
      inp = global_avg_pool(inp)
    else:
      inp, n_features_inp = prepare_collapsed_input_and_dropout(inputs, dropout)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", [n_features_inp, n_classes], l2, tower_setup)
      b = self.create_bias_variable("b", [n_classes], tower_setup)
      y_ref = tf.cast(targets, tf.int64)
      y_pred = tf.matmul(inp, W) + b
      self.outputs = [tf.nn.softmax(y_pred, -1, 'softmax')]
      errors = tf.not_equal(tf.argmax(y_pred, 1), y_ref)
      errors = tf.reduce_sum(tf.cast(errors, tower_setup.dtype))
      self.measures['errors'] = errors

      if loss == "ce":
        cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=y_pred, labels=y_ref, name='cross_entropy_per_example')
        self.loss = tf.reduce_sum(cross_entropy_per_example, name='cross_entropy_sum')
      else:
        assert False, "Unknown loss " + loss

      self.add_scalar_summary(self.loss, "loss")


class Clustering(Layer):
  output_layer = True

  #note: we need to have n_classes in the signature of an output layer, but will not actually use it
  def __init__(self, name, inputs, targets, n_classes, n_clusters, tower_setup, use_complete_batch, use_SPN,
               exclude_zeros_from_loss, original_labels=None, margin=2.0, dropout=0.0, l2=L2_DEFAULT):
    super(Clustering, self).__init__()
    self.measures = {}
    inp, n_features_inp = prepare_collapsed_input_and_dropout(inputs, dropout)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", [n_features_inp, n_clusters], l2, tower_setup)
      b = self.create_bias_variable("b", [n_clusters], tower_setup)
      y_pred = tf.matmul(inp, W) + b
      y_pred = tf.nn.softmax(y_pred, -1, 'softmax')
      self.outputs = [y_pred]

      summ = tf.summary.histogram("softmax", y_pred)
      self.summaries.append(summ)

      if use_complete_batch:
        #original_labels = targets
        curr = y_pred
        batch_size = smart_shape(curr)[0]
        curr1 = curr[:, tf.newaxis, :]
        curr2 = curr[tf.newaxis, :, :]

        curr1 = tf.transpose(curr1, perm=[1, 0, 2])
        curr1_big = tf.tile(curr1, [batch_size, 1, 1])
        curr1_big = tf.transpose(curr1_big, perm=[1, 0, 2])
        curr2_big = tf.tile(curr2, [batch_size, 1, 1])

        boolean_target = tf.cast(tf.ones([batch_size, batch_size]), tf.bool)
        #### Following extracts one sided + diagonal, but for now we are using both sided as concat is non-symmetric
        #### In future may want to remove the diagonal
        # boolean_target = tf.matrix_band_part(boolean_target, -1, 0),tf.bool

        y_pred0 = tf.boolean_mask(curr1_big, boolean_target)
        y_pred1 = tf.boolean_mask(curr2_big, boolean_target)

        if not use_SPN:
          targets1 = targets[:, tf.newaxis]
          targets2 = targets[tf.newaxis, :]
          whole_targets = tf.cast(tf.equal(targets1, targets2), tf.int32)
          targets = tf.boolean_mask(whole_targets, boolean_target)
      else:
        y_pred0 = y_pred[0::2]
        y_pred1 = y_pred[1::2]
        targets = targets[::2]

      if original_labels is not None:
        cluster_ids = tf.argmax(y_pred, axis=-1)
        self.measures[Constants.CLUSTER_IDS] = [cluster_ids]
        self.measures[Constants.ORIGINAL_LABELS] = [original_labels]

      #y_pred0_stopped = tf.stop_gradient(y_pred0)
      #y_pred1_stopped = tf.stop_gradient(y_pred1)

      def kl(x, y):
        epsilon = tf.constant(1e-8, tf.float32)
        x += epsilon
        y += epsilon
        return tf.reduce_sum(x * tf.log(x / y), axis=-1)

      kl1 = kl(y_pred0, y_pred1)
      kl2 = kl(y_pred1, y_pred0)
      #kl1 = kl(y_pred0_stopped, y_pred1)
      #kl2 = kl(y_pred1_stopped, y_pred0)

      def Lh(x):
        return tf.nn.relu(margin - x)
        # return tf.nn.softplus(-x)
      pos_loss = kl1 + kl2
      neg_loss = Lh(kl1) + Lh(kl2)
      loss = tf.where(tf.cast(targets, tf.bool), pos_loss, neg_loss)
      if exclude_zeros_from_loss:
        norm_factor = tf.maximum(tf.count_nonzero(loss, dtype=tf.float32), 1.0)
        loss /= norm_factor
        loss *= tf.cast(smart_shape(inp)[0], tf.float32)
      elif use_complete_batch:
        loss /= tf.cast(smart_shape(inp)[0], tf.float32)
      self.loss = tf.reduce_sum(loss)
      self.add_scalar_summary(self.loss, "loss")
