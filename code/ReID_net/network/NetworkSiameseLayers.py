import tensorflow as tf
import numpy

from ReID_net.datasets.Util.Util import smart_shape
from .NetworkLayers import Layer
from .Util_Network import prepare_input


class SiameseMerge(Layer):
  def __init__(self, name, inputs, tower_setup, merge_type):
    super(SiameseMerge, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    if merge_type == "concat":
      out = tf.reshape(curr, [-1, n_features_inp * 2])
    elif merge_type == "add":
      part1 = curr[::2, :]
      part2 = curr[1::2, :]
      out = part1 + part2
    elif merge_type == "subtract":
      part1 = curr[::2, :]
      part2 = curr[1::2, :]
      out = part1 - part2
    elif merge_type == "abs_subtract":
      part1 = curr[::2, :]
      part2 = curr[1::2, :]
      out = tf.abs(part1 - part2)
    else:
      out = curr
    self.outputs = [out]


class ExpandedSiameseMerge(Layer):
  def __init__(self, name, inputs, targets, tower_setup, merge_type):
    super(ExpandedSiameseMerge, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    size = smart_shape(curr)[0]

    def Expand(idx):
      anchor = curr[idx, :]
      anchor_class = targets[idx]
      classes,classes_ids = tf.unique(targets)
      anchor_class_id = classes_ids[idx]
      class_division = tf.cast(tf.equal(targets, anchor_class), tf.int32) - tf.cast(tf.equal(list(range(0,size)),idx),tf.int32)
      partitioned_output = tf.dynamic_partition(curr, class_division, 2)
      #partitioned_targets = tf.dynamic_partition(targets, class_division, 2)

      # Positives
      positives = partitioned_output[1]

      size_positives = smart_shape(positives)[0]
      anchor_positive_repmat = tf.reshape(tf.tile(anchor,[size_positives]),[size_positives,-1])
      positives_combined = tf.concat((anchor_positive_repmat,positives),1)
      new_targets_positive = tf.ones([smart_shape(positives_combined)[0]],dtype=tf.int32)

      # Negatives
      negative_size = smart_shape(classes)[0]

      def Get_negatives(neg_idx):
        curr_neg_class = classes[neg_idx]
        neg_class_division = tf.cast(tf.equal(targets, curr_neg_class), tf.int32)
        neg_partitioned_output = tf.dynamic_partition(curr, neg_class_division, 2)
        negative_set = neg_partitioned_output[1]
        size_negative_set = smart_shape(negative_set)[0]
        random_negative_idx = tf.random_shuffle(tf.range(1, size_negative_set))[0]
        random_negative = negative_set[random_negative_idx,:]
        return random_negative

      looper = tf.range(0, anchor_class_id)
      iter_val = tf.minimum(anchor_class_id+1,negative_size)
      # looper = tf.range(0, idx)
      # iter_val = tf.minimum(idx + 1, negative_size)
      looper = tf.concat([looper,tf.range(iter_val,negative_size)],0)

      negatives = tf.map_fn(Get_negatives, looper, dtype=tf.float32)
      size_negatives = smart_shape(negatives)[0]
      anchor_negative_repmat = tf.reshape(tf.tile(anchor, [size_negatives]), [size_negatives, -1])
      negatives_combined = tf.concat((anchor_negative_repmat,negatives),1)
      new_targets_negative = tf.zeros([smart_shape(negatives_combined)[0]],dtype=tf.int32)

      all_combined = tf.concat((positives_combined,negatives_combined),0)
      new_targets_combined = tf.concat((new_targets_positive,new_targets_negative),0)

      return all_combined, new_targets_combined

    expanded, new_targets = tf.map_fn(Expand, tf.range(0, size), dtype=(tf.float32, tf.int32))

    if merge_type == "concat":
      expanded = tf.reshape(expanded, [-1, n_features_inp * 2])
    elif merge_type == "add":
      part1 = expanded[::2, :]
      part2 = expanded[1::2, :]
      expanded = part1 + part2
    elif merge_type == "subtract":
      part1 = expanded[::2, :]
      part2 = expanded[1::2, :]
      expanded = part1 - part2
    elif merge_type == "abs_subtract":
      part1 = expanded[::2, :]
      part2 = expanded[1::2, :]
      expanded = tf.abs(part1 - part2)
    else:
      expanded = expanded

    new_targets = tf.reshape(new_targets, [-1])

    debug = 0
    if debug:
      # Print - debug example code
      def test(a):
        print(a)
        return numpy.array([5], dtype="int32")

      t, = tf.py_func(test, [smart_shape(new_targets)], [tf.int32])
      with tf.control_dependencies([t]):
        expanded = tf.identity(expanded)

    self.outputs = [expanded]
    self.out_labels = new_targets


class CompleteSiameseMerge(Layer):
  def __init__(self, name, inputs, targets, tower_setup, merge_type):
    super(CompleteSiameseMerge, self).__init__()
    curr, n_features_inp = prepare_input(inputs)

    batch_size = smart_shape(curr)[0]
    curr1 = curr[:,tf.newaxis,:]
    curr2 = curr[tf.newaxis,:,:]

    if merge_type == "concat":
      # curr1_big = tf.reshape(tf.tile(curr1, [batch_size]), [batch_size, -1])
      # curr2_big = tf.reshape(tf.tile(curr2, [batch_size]), [batch_size, -1])

      curr1 = tf.transpose(curr1, perm=[1, 0, 2])
      curr1_big = tf.tile(curr1, [batch_size,1,1])
      curr1_big = tf.transpose(curr1_big, perm=[1, 0, 2])
      curr2_big = tf.tile(curr2, [batch_size,1,1])

      whole_mat = tf.concat((curr1_big,curr2_big),2)
    elif merge_type == "add":
      whole_mat = curr1 + curr2
    elif merge_type == "subtract":
      whole_mat = curr1 - curr2
    elif merge_type == "abs_subtract":
      whole_mat = tf.abs(curr1 - curr2)
    else:
      raise ValueError("No correct merge type")

    targets1 = targets[:,tf.newaxis]
    targets2 = targets[tf.newaxis,:]
    whole_targets = tf.cast(tf.equal(targets1,targets2),tf.int32)

    boolean_target = tf.cast(tf.ones([batch_size,batch_size]),tf.bool)
    #### Following extracts one sided + diagonal, but for now we are using both sided as concat is non-symmetric
    #### In future may want to remove the diagonal
    # boolean_target = tf.matrix_band_part(boolean_target, -1, 0),tf.bool
    targets_list = tf.boolean_mask(whole_targets,boolean_target)
    vectors_list = tf.boolean_mask(whole_mat,boolean_target)

    debug = 0
    if debug:
      # Print - debug example code
      def test(a):
        print(a)
        return numpy.array([5], dtype="int32")

      t, = tf.py_func(test, [smart_shape(targets_list)], [tf.int32])
      with tf.control_dependencies([t]):
        targets_list = tf.identity(targets_list)

      t, = tf.py_func(test, [smart_shape(vectors_list)], [tf.int32])
      with tf.control_dependencies([t]):
        targets_list = tf.identity(targets_list)

    self.outputs = [vectors_list]
    self.out_labels = targets_list