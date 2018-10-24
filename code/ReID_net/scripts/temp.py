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
