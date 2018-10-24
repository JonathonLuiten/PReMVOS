import tensorflow as tf
import numpy as np
import ReID_net.Constants as Constants
from ReID_net.Log import log
from ReID_net.Util import average_gradients, clip_gradients


class Trainer(object):
  def __init__(self, config, train_network, test_network, global_step, session):
    self.profile = config.bool("profile", False)
    self.add_grad_checks = config.bool("add_grad_checks", False)
    self.add_numerical_checks = config.bool("add_numerical_checks", False)
    self.measures = config.unicode_list("measures", [])
    self.opt_str = config.str("optimizer", "adam").lower()
    self.train_network = train_network
    self.test_network = test_network
    self.session = session
    self.global_step = global_step
    self.validation_step_number = 0
    self.gradient_clipping = config.float("gradient_clipping", -1.0)
    self.optimizer_exclude_prefix = config.str("optimizer_exclude_prefix", "")
    self.learning_rates = config.int_key_dict("learning_rates")
    self.recursive_training = config.bool(Constants.RECURSIVE_TRAINING, False)
    assert 1 in self.learning_rates, "no initial learning rate specified"
    self.curr_learning_rate = self.learning_rates[1]
    self.lr_var = tf.placeholder(config.dtype, shape=[], name="learning_rate")
    self.loss_scale_var = tf.placeholder_with_default(1.0, shape=[], name="loss_scale")
    self.opt, self.reset_opt_op = self.create_optimizer(config)
    grad_norm = None
    if train_network is not None:
      if train_network.use_partialflow:
        self.prepare_partialflow()
        self.step_op = tf.no_op("step")
      else:
        self.step_op, grad_norm = self.create_step_op()
      if len(self.train_network.update_ops) == 0:
        self.update_ops = []
      else:
        self.update_ops = self.train_network.update_ops
      if self.add_numerical_checks:
        self.update_ops.append(tf.add_check_numerics_ops())
      self.train_targets = self.train_network.raw_labels
      self.train_inputs = self.train_network.inputs
      self.train_network_ys = self.train_network.y_softmax
      if self.train_network_ys is not None and self.train_targets is not None:
        self.train_network_ys = self._adjust_results_to_targets(self.train_network_ys, self.train_targets)
    else:
      self.step_op = None
      self.update_ops = None
    self.summary_writer, self.summary_op, self.summary_op_test = self.init_summaries(config, grad_norm)

    if test_network is not None:
      self.test_targets = self.test_network.raw_labels
      self.test_inputs = self.test_network.inputs
      self.test_network_ys = self.test_network.y_softmax
      if self.test_network_ys is not None and self.test_targets is not None:
        self.test_network_ys = self._adjust_results_to_targets(self.test_network_ys, self.test_targets)

  def create_optimizer(self, config):
    momentum = config.float("momentum", 0.9)
    if self.opt_str == "sgd_nesterov":
      return tf.train.MomentumOptimizer(self.lr_var, momentum, use_nesterov=True), None
    elif self.opt_str == "sgd_momentum":
      return tf.train.MomentumOptimizer(self.lr_var, momentum), None
    elif self.opt_str == "sgd":
      return tf.train.GradientDescentOptimizer(self.lr_var), None
    elif self.opt_str == "adam":
      opt = tf.train.AdamOptimizer(self.lr_var)
      all_vars = tf.global_variables()
      opt_vars = [v for v in all_vars if "Adam" in v.name]
      reset_opt_op = tf.variables_initializer(opt_vars, "reset_optimizer")
      return opt, reset_opt_op
    elif self.opt_str == "yellowfin":
      from ReID_net.external.yellowfin import YFOptimizer
      return YFOptimizer(sparsity_debias=False), None
    elif self.opt_str == "none":
      return None, None
    else:
      assert False, ("unknown optimizer", self.opt_str)

  def reset_optimizer(self):
    assert self.opt_str == "adam", "reset not implemented for other optimizers yet"
    assert self.reset_opt_op is not None
    self.session.run(self.reset_opt_op)

  def prepare_partialflow(self):
    sm = self.train_network.graph_section_manager
    losses = self.train_network.losses
    regularizers = self.train_network.regularizers
    assert len(losses) == 1
    assert len(regularizers) == 1
    loss = losses[0] + tf.add_n(regularizers[0])
    loss *= self.loss_scale_var
    sm.add_training_ops(self.opt, loss, verbose=False, global_step=self.global_step)
    sm.prepare_training()
    #for sec in self.train_network.graph_sections:
    #  print sec.get_tensors_to_feed()
    #for sec in self.train_network.graph_sections:
    #  print sec.get_tensors_to_cache()

  def create_step_op(self):
    if self.opt is None:
      return tf.no_op("dummy_step_op"), None

    losses, regularizers, setups = self.train_network.losses, self.train_network.regularizers, \
        self.train_network.tower_setups
    assert len(losses) == len(regularizers)
    assert all(len(regularizers[0]) == len(x) for x in regularizers)
    regularizers = [tf.add_n(tower_regularizers) if len(tower_regularizers) > 0 else tf.constant(0, tf.float32) for
                    tower_regularizers in regularizers]
    losses_with_regularizers = [l + r for l, r in zip(losses, regularizers)]
    tower_grads = []
    for l, s in zip(losses_with_regularizers, setups):
      gpu_str = "/gpu:" + str(s.gpu)
      with tf.device(gpu_str), tf.name_scope("tower_gpu_" + str(s.gpu) + "_opt"):
        var_list = (
          tf.trainable_variables() +
          tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        if self.optimizer_exclude_prefix != "":
          to_remove = [x.name for x in var_list if x.name.startswith(self.optimizer_exclude_prefix)]
          if len(to_remove) > 0:
            print("excluding", to_remove, "from optimization, since they start with prefix", \
              self.optimizer_exclude_prefix, file=log.v1)
            var_list = [x for x in var_list if not x.name.startswith(self.optimizer_exclude_prefix)]
          else:
            print("warning, optimizer_exclude_prefix=", self.optimizer_exclude_prefix, "is specified," \
                             " but no variable with this prefix is present in the model", file=log.v1)
        grads_raw = self.opt.compute_gradients(l, var_list=var_list)
        #filter out gradients w.r.t. disconnected variables
        grads_filtered = [g for g in grads_raw if g[0] is not None]
        tower_grads.append(grads_filtered)

    with tf.device(setups[0].variable_device):
      if len(losses) == 1:
        grads = tower_grads[0]
      else:
        # average the gradients over the towers
        grads = average_gradients(tower_grads)

      if self.add_grad_checks:
        grads = [(tf.check_numerics(x[0], x[1].name), x[1]) for x in grads]

      #grad clipping
      if self.gradient_clipping != -1:
        grads, norm = clip_gradients(grads, self.gradient_clipping)
      else:
        norm = None

      step_op = self.opt.apply_gradients(grads, global_step=self.global_step)
    return step_op, norm

  def init_summaries(self, config, grad_norm=None):
    summdir = config.dir("summary_dir", "summaries")
    model = config.str("model")
    summdir += model + "/"
    tf.gfile.MakeDirs(summdir)
    summary_writer = tf.summary.FileWriter(summdir, self.session.graph)
    summary_op = None
    summary_op_test = None
    if config.bool("write_summaries", True):
      if self.train_network is not None and len(self.train_network.summaries) > 0:
        # better do not merge ALL summaries, since otherwise we get summaries from different networks
        # and might execute (parts of) the test network while training
        # self.summary_op = tf.merge_all_summaries()
        # atm we only collect summaries from the train network
        if grad_norm is None:
          summary_op = tf.summary.merge(self.train_network.summaries)
        else:
          #grad_norm = tf.Print(grad_norm, [grad_norm], "grad_norm")
          grad_norm_summary = tf.summary.scalar("grad_norm", grad_norm)
          summary_op = tf.summary.merge(self.train_network.summaries + [grad_norm_summary])
      if self.test_network is not None and len(self.test_network.summaries) > 0:
        summary_op_test = tf.summary.merge(self.test_network.summaries)
    return summary_writer, summary_op, summary_op_test

  def get_options(self):
    if self.profile:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      return run_options, run_metadata
    else:
      return None, None

  # for profiling
  def handle_run_metadata(self, metadata, step):
    if metadata is None:
      return
    if not self.profile:
      return
    self.summary_writer.add_run_metadata(metadata, "profile%d" % step, step)
    #leave a few steps for warmup and then write out at the 10th step
    if step == 10:
      from tensorflow.python.client import timeline
      tl = timeline.Timeline(metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open('timeline.json', 'w') as f:
        f.write(ctf)

  def validation_step(self, _):
    ops = [self.test_network.loss_summed, self.test_network.measures_accumulated, self.test_network.n_imgs]
    if 'clicks' in self.measures:
      ops.append(self.test_network.tags)

    if self.recursive_training:
      ops.append(self.test_network.tags)
      ops.append(self.test_network_ys)
      ops.append(self.test_targets)

    if self.summary_op_test is not None:
      ops.append(self.summary_op_test)

    res = self.session.run(ops)
    if self.summary_op_test is not None:
      summary_str = res[-1]
      res = res[:-1]
      self.summary_writer.add_summary(summary_str, global_step=self.validation_step_number)
      self.validation_step_number += 1

    if len(res) > 4:
      loss_summed, measures_accumulated, n_imgs, tags, ys_val, targets = res
      ys_argmax_val = np.argmax(ys_val, axis=-1)
      return loss_summed, measures_accumulated, n_imgs, tags, ys_argmax_val, targets
    elif len(res) > 3:
      loss_summed, measures_accumulated, n_imgs, tags = res
      measures_accumulated[Constants.CLICKS] = tags
    else:
      loss_summed, measures_accumulated, n_imgs = res

    return loss_summed, measures_accumulated, n_imgs

  def adjust_learning_rate(self, epoch, learning_rate=None):
    if learning_rate is None:
      key = max([k for k in list(self.learning_rates.keys()) if k <= epoch + 1])
      new_lr = self.learning_rates[key]
    else:
      new_lr = learning_rate
    if self.curr_learning_rate != new_lr:
      print("changing learning rate to", new_lr, file=log.v1)
      self.curr_learning_rate = new_lr

  def train_step(self, epoch, feed_dict=None, loss_scale=1.0, learning_rate=None):
    self.adjust_learning_rate(epoch, learning_rate)
    if feed_dict is None:
      feed_dict = {}
    else:
      feed_dict = feed_dict.copy()
    feed_dict[self.lr_var] = self.curr_learning_rate
    feed_dict[self.loss_scale_var] = loss_scale

    ops = self.update_ops + [self.global_step, self.step_op, self.train_network.loss_summed,
                             self.train_network.measures_accumulated, self.train_network.n_imgs]

    if self.recursive_training:
      ops.append(self.train_network.tags)
      ops.append(self.train_network_ys)
      ops.append(self.train_targets)
    elif Constants.CLICKS in self.measures:
      ops.append(self.train_network.tags)

    if self.summary_op is not None:
      ops.append(self.summary_op)

    if self.train_network.use_partialflow:
      res = self.train_network.graph_section_manager.run_full_cycle(
        self.session, fetches=ops, basic_feed=feed_dict)
      run_metadata = None
    else:
      run_options, run_metadata = self.get_options()
      res = self.session.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

    #remove update outputs
    res = res[len(self.update_ops):]
    step = res[0]

    if self.summary_op is not None:
      summary_str = res[-1]
      res = res[:-1]
      self.summary_writer.add_summary(summary_str, step)

    self.handle_run_metadata(run_metadata, step)

    if len(res) > 6:
      _, _, loss_summed, measures_accumulated, n_imgs, tags, ys_val, targets = res
      ys_argmax_val = np.argmax(ys_val, axis=-1)
      return loss_summed, measures_accumulated, n_imgs, tags, ys_argmax_val, targets
    elif len(res) == 6:
      _, _, loss_summed, measures_accumulated, n_imgs, tags = res
      measures_accumulated[Constants.CLICKS] = tags
    else:
      _, _, loss_summed, measures_accumulated, n_imgs = res

    return loss_summed, measures_accumulated, n_imgs

  def _adjust_results_to_targets(self, y_softmax, targets):
    # scale it up!
    return tf.image.resize_images(y_softmax, tf.shape(targets)[1:3])
