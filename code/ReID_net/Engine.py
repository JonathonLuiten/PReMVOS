import glob
import time
import numpy as np
import os
import shlex
import subprocess

import tensorflow as tf
from tensorflow.contrib.framework import list_variables

import ReID_net.Constants as Constants
import ReID_net.Measures as Measures
from ReID_net.Forwarding import IterativeImageForwarder, OneshotForwarder, DatasetSpeedtestForwarder
from ReID_net.Forwarding.CMC_Validator import do_cmc_validation
from ReID_net.Log import log
from ReID_net.Trainer import Trainer
from ReID_net.Util import load_wider_or_deeper_mxnet_model
from ReID_net.datasets.Forward import forward, oneshot_forward, online_forward, offline_forward, interactive_forward, \
  forward_clustering, forward_reid
from ReID_net.datasets.Loader import load_dataset
from ReID_net.network.Network import Network
from tensorflow.contrib import slim


class Engine(object):
  def __init__(self, config, session=None):
    self.config = config
    self.tfdbg = config.bool("tfdbg", False)
    self.dataset = config.str("dataset").lower()
    try:
      self.load_init = config.str("load_init", "")
      if self.load_init == "":
        self.load_init = []
      else:
        self.load_init = [self.load_init]
    except TypeError:
      self.load_init = config.unicode_list("load_init", [])
    self.load = config.str("load", "")
    self.task = config.str("task", "train")
    self.use_partialflow = config.bool("use_partialflow", False)
    self.store_detections_after_each_epoch = config.bool("store_detections_after_each_epoch", False)
    self._detection_file = None
    self.num_epochs = config.int("num_epochs", 1000)
    self.model = config.str("model")
    self.model_base_dir = config.dir("model_dir", "models")
    self.model_dir = self.model_base_dir + self.model + "/"
    self.save = config.bool("save", True)
    self.do_oneshot_or_online = self.task in ("oneshot_forward", "oneshot", "oneshot_instance", "online")
    if self.do_oneshot_or_online:
      assert config.int("batch_size_eval", 1) == 1
    self.need_train = self.task in ("train", "train_no_val","forward_train", "forward_clustering_train",
                                    "dataset_speedtest", Constants.ITERATIVE_FORWARD, Constants.ONESHOT_INTERACTIVE) \
        or self.do_oneshot_or_online
    self.use_pre_saved_data = self.config.bool("use_pre_saved_data", False)

    if config.bool('set_visible_gpus',False):
      self.set_visible_gpus()
    elif config.bool('use_free_gpu',False):
      self.use_free_gpu()

    if session is None:
      sess_config = tf.ConfigProto(allow_soft_placement=True)
      sess_config.gpu_options.allow_growth = True
      # self.session = tf.InteractiveSession(config=sess_config)
      # self.session = tf.Session(graph=tf.Graph(),config=sess_config)
      self.session = tf.Session(config=sess_config)
      if self.tfdbg:
        from tensorflow.python import debug as tf_debug
        from ReID_net.Util import debug_is_zero
        self.session = tf_debug.LocalCLIDebugWrapperSession(self.session, thread_name_filter="MainThread$")
                                            #dump_root="/fastwork/" + username() + "/mywork/data/tmp_tfdbg/")
        self.session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        self.session.add_tensor_filter("is_zero", debug_is_zero)
    else:
      self.session = session
    self.coordinator = tf.train.Coordinator()
    if self.task == "dataset_speedtest":
      self.train_data = load_dataset(config, "train", self.session, self.coordinator)
      return

    self.need_val = self.task not in ["train_no_val"]

    if self.use_pre_saved_data or not self.need_val:
      self.valid_data = None
    else:
      self.valid_data = load_dataset(config, "valid", self.session, self.coordinator)
    if self.need_train:
      self.train_data = load_dataset(config, "train", self.session, self.coordinator)
    self.cmc_validation = self.dataset in ("cuhk", "cuhk03")
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.start_epoch = 0
    reuse_variables = None
    if self.need_train:
      freeze_batchnorm = config.bool("freeze_batchnorm", False)
      self.train_network = Network(config, self.train_data, self.global_step, training=True,
                                   use_partialflow=self.use_partialflow, freeze_batchnorm=freeze_batchnorm,
                                   name="trainnet")
      reuse_variables = True
    else:
      self.train_network = None
    if not self.use_pre_saved_data and self.valid_data is not None:
      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        self.test_network = Network(config, self.valid_data, self.global_step, training=False,
                                    use_partialflow=False, freeze_batchnorm=True, name="ReID_net")
        print("number of parameters:", "{:,}".format(self.test_network.n_params))
    else:
      self.test_network = None
    self.trainer = Trainer(config, self.train_network, self.test_network, self.global_step, self.session)
    max_saves_to_keep = config.int("max_saves_to_keep", 0)
    self.saver = tf.train.Saver(max_to_keep=max_saves_to_keep, pad_step_number=True)
    tf.global_variables_initializer().run(session=self.session)
    tf.local_variables_initializer().run(session=self.session)
    tf.train.start_queue_runners(self.session)
    self.load_init_savers = None

    self.recursive_training = config.bool(Constants.RECURSIVE_TRAINING, False)
    if not self.do_oneshot_or_online:
      self.try_load_weights()
      # put this in again later
      # self.session.graph.finalize()

  def set_visible_gpus(self):

    gpus = self.config.int_list("gpus")
    gpu_list_string = ','.join([str(i) for i in gpus])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list_string

  def use_free_gpu(self):

    # Find GPU with most free memory
    command = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'
    result = subprocess.check_output(shlex.split(command),universal_newlines=True)
    # result = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, universal_newlines=True).stdout
    free_memories = [int(line) for line in result.split('\n') if line]
    gpu_id = np.argmax(free_memories)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

  def _create_load_init_saver(self, filename):
    if self.load != "":
      return None
    if len(glob.glob(self.model_dir + self.model + "-*.index")) > 0:
      return None
    if filename == "" or filename.endswith(".pickle"):
      return None

    vars_and_shapes_file = [x for x in list_variables(filename) if x[0] != "global_step"]
    vars_file = [x[0] for x in vars_and_shapes_file]
    vars_to_shapes_file = {x[0]: x[1] for x in vars_and_shapes_file}
    vars_model = tf.global_variables()
    assert all([x.name.endswith(":0") for x in vars_model])
    vars_intersection = [x for x in vars_model if x.name[:-2] in vars_file]
    vars_missing_in_graph = [x for x in vars_model if x.name[:-2] not in vars_file and "Adam" not in x.name and
                             "beta1_power" not in x.name and "beta2_power" not in x.name]
    if len(vars_missing_in_graph) > 0:
      print("the following variables will not be initialized since they are not present in the " \
                       "initialization model", [v.name for v in vars_missing_in_graph])

    var_names_model = [x.name for x in vars_model]
    vars_missing_in_file = [x for x in vars_file if x + ":0" not in var_names_model
                            and "RMSProp" not in x and "Adam" not in x and "Momentum" not in x]
    if len(vars_missing_in_file) > 0:
      print("the following variables will not be loaded from the file since they are not present in the " \
                       "graph", vars_missing_in_file)

    vars_shape_mismatch = [x for x in vars_intersection if x.shape.as_list() != vars_to_shapes_file[x.name[:-2]]]
    if len(vars_shape_mismatch) > 0:
      print("the following variables will not be loaded from the file since the shapes in the graph and in" \
                       " the file don't match:", [(x.name, x.shape) for x in vars_shape_mismatch
                                                  if "Adam" not in x.name])
      vars_intersection = [x for x in vars_intersection if x not in vars_shape_mismatch]
    return tf.train.Saver(var_list=vars_intersection)

  def try_load_weights(self):
    fn = None
    if self.load != "":
      fn = self.load.replace(".index", "")
    else:
      files = sorted(glob.glob(self.model_dir + self.model + "-*.index"))
      if len(files) > 0:
        fn = files[-1].replace(".index", "")

    if fn is not None:
      print("loading model from", fn)

      # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ReID_net')
      varlist = slim.get_variables()
      # print (np.unique(np.array([var.name.split('/')[0] for var in varlist])))
      good_list = ['conv0','conv1','fc1','fc2','outputTriplet', 'res0', 'res1', 'res10', 'res11','res12', 'res13',
                   'res14', 'res15', 'res16', 'res2', 'res3', 'res4', 'res5','res6', 'res7', 'res8', 'res9']
      varlist = [var for var in varlist if var.name.split('/')[0] in good_list]
      self.saver = tf.train.Saver(pad_step_number=True, var_list=varlist)

      self.saver.restore(self.session, fn)
      if self.model == fn.split("/")[-2]:
        self.start_epoch = int(fn.split("-")[-1])
        print("starting from epoch", self.start_epoch + 1)
    else:
      if self.load_init_savers is None:
        self.load_init_savers = [self._create_load_init_saver(x) for x in self.load_init]
      assert len(self.load_init) == len(self.load_init_savers)
      for fn, load_init_saver in zip(self.load_init, self.load_init_savers):
        if fn.endswith(".pickle"):
          print("trying to initialize model from wider-or-deeper mxnet model", fn)
          load_wider_or_deeper_mxnet_model(fn, self.session)
        else:
          print("initializing model from", fn)
          assert load_init_saver is not None
          load_init_saver.restore(self.session, fn)

  def reset_optimizer(self):
    self.trainer.reset_optimizer()

  def run_epoch(self, step_fn, data, epoch, train):
    loss_total = 0.0
    n_imgs_per_epoch = data.num_examples_per_epoch()
    measures_accumulated = {}
    n_imgs_processed = 0
    if hasattr(data, "ignore_classes"):
      ignore_classes = data.ignore_classes
    else:
      ignore_classes = None

    while n_imgs_processed < n_imgs_per_epoch:
      start = time.time()
      res = step_fn(epoch)
      if len(res) > 3:
        loss_summed, measures, n_imgs, tags, ys_armax_val, targets = res
        if hasattr(data, "set_output_as_old_label"):
          data.set_output_as_old_label(tags, ys_armax_val, epoch, targets)
      else:
        loss_summed, measures, n_imgs = res

      loss_total += loss_summed

      #special handling for detection storing
      self._maybe_store_detections(epoch, train, measures)

      measures_accumulated = Measures.calc_measures_sum(measures_accumulated, measures)

      n_imgs_processed += n_imgs

      loss_avg = loss_summed / n_imgs
      #do not compute expensive measures here, since it's not the final result for the epoch
      measures_avg = Measures.calc_measures_avg(measures, n_imgs, ignore_classes, for_final_result=False)
      end = time.time()
      elapsed = end - start

      # TODO: Print proper averages for the measures
      print(n_imgs_processed, '/', n_imgs_per_epoch, loss_avg, measures_avg, "elapsed", elapsed)
    loss_total /= max(n_imgs_processed, 1)
    measures_accumulated = Measures.calc_measures_avg(measures_accumulated, n_imgs_processed, ignore_classes,
                                                      for_final_result=True)
    self._maybe_finalize_detections(train)
    return loss_total, measures_accumulated

  def train(self):
    assert self.need_train
    print("starting training")
    for epoch in range(self.start_epoch, self.num_epochs):
      start = time.time()
      train_loss, train_measures = self.run_epoch(self.trainer.train_step, self.train_data, epoch, train=True)

      if self.cmc_validation:
        valid_loss, valid_measures = do_cmc_validation(self, self.test_network, self.valid_data)
      elif self.recursive_training:
        for valid_round in range(3):
          valid_loss, valid_measures = self.run_epoch(self.trainer.validation_step, self.valid_data, epoch, train=False)
          valid_error_string = Measures.get_error_string(valid_measures, "valid")
          print("Validation ", valid_round, ": ", valid_error_string)
        if hasattr(self.valid_data, "clear_data_dict"):
          self.valid_data.clear_data_dict()
      elif self.valid_data is not None:
        valid_loss, valid_measures = self.run_epoch(self.trainer.validation_step, self.valid_data, epoch, train=False)
      else:
        valid_loss = 0.0
        valid_measures = {}

      end = time.time()
      elapsed = end - start
      train_error_string = Measures.get_error_string(train_measures, "train")
      valid_error_string = Measures.get_error_string(valid_measures, "valid")
      print("epoch", epoch + 1, "finished. elapsed:", "%.5f" % elapsed, "train_score:",
          "%.5f" % train_loss, train_error_string, "valid_score:", valid_loss, valid_error_string)
      print("epoch", epoch + 1, "finished. elapsed:", "%.5f" % elapsed, "train_score:",
          "%.5f" % train_loss, train_error_string, "valid_score:", valid_loss, valid_error_string,
            file=open("/home/luiten/vision/youtubevos/ReID_net/logs/"+self.model+".txt", "a"))
      if self.save:
        self.save_model(epoch + 1)
        if hasattr(self.train_data, "save_masks"):
          self.train_data.save_masks(epoch + 1)

  def eval(self):
    start = time.time()
    if self.cmc_validation:
      valid_loss, measures = do_cmc_validation(self, self.test_network, self.valid_data)
    else:
      valid_loss, measures = self.run_epoch(self.trainer.validation_step, self.valid_data, 0, train=False)
    end = time.time()
    elapsed = end - start
    valid_error_string = Measures.get_error_string(measures, "valid")
    print("eval finished. elapsed:", elapsed, "valid_score:", valid_loss, valid_error_string)

  def run(self):
    if self.task == "train" or self.task == "train_no_val":
      self.train()
    elif self.task == "eval":
      self.eval()
    elif self.task in ("forward", "forward_train"):
      if self.task == "forward_train":
        network = self.train_network
        data = self.train_data
      else:
        network = self.test_network
        data = self.valid_data
      save_logits = self.config.bool("save_logits", False)
      save_results = self.config.bool("save_results", True)
      forward(self, network, data, self.dataset, save_results=save_results, save_logits=save_logits)
    elif self.task == "forward_clustering":
      network = self.test_network
      data = self.valid_data
      forward_clustering(self, network, data)
    elif self.task == "forward_clustering_train":
      network = self.train_network
      data = self.train_data
      forward_clustering(self, network, data)
    elif self.task == "forward_ReID":
      network = self.test_network
      data = self.valid_data
      forward_reid(self, network, data)
    elif self.task in (Constants.FORWARD_INTERACTIVE, Constants.FORWARD_RECURSIVE):
      network = self.test_network
      data = self.valid_data
      save_logits = self.config.bool("save_logits", False)

      if self.task == Constants.FORWARD_INTERACTIVE:
        save_results = self.config.bool("save_results", True)
      else:
        save_results = self.config.bool("save_results", True)
      interactive_forward(self, network, data, save_results, save_logits, self.task)

    elif self.task == Constants.ONESHOT_INTERACTIVE:
      oneshot_forwarder = OneshotForwarder.InteractiveOneshotForwarder(self)
      oneshot_forwarder.forward(network=self.test_network, data=self.valid_data,
                                save_results=True)
    elif self.task == Constants.ITERATIVE_FORWARD:
      save_logits = self.config.bool("save_logits", False)
      save_results = self.config.bool("save_results", True)
      iterative_forwarder = IterativeImageForwarder.IterativeImageForwarder(self)
      iterative_forwarder.forward(network=self.test_network, data=self.valid_data,
                                  save_results=save_results, save_logits=save_logits)
    elif self.task == "dataset_speedtest":
      forwarder = DatasetSpeedtestForwarder.DatasetSpeedtestForwarder(self)
      forwarder.forward(network=None, data=self.train_data)
    elif self.do_oneshot_or_online:
      save_logits = self.config.bool("save_logits", False)
      save_results = self.config.bool("save_results", False)
      if self.task == "oneshot":
        oneshot_forward(self, save_results=save_results, save_logits=save_logits)
      elif self.task == "online":
        online_forward(self, save_results=save_results, save_logits=save_logits)
      else:
        offline_forward(self, save_results=save_results, save_logits=save_logits)
        assert self.task == "offline"
    else:
      assert False, "Unknown task " + str(self.task)

  def save_model(self, epoch):
    tf.gfile.MakeDirs(self.model_dir)
    self.saver.save(self.session, self.model_dir + self.model, epoch)

  def _maybe_store_detections(self, epoch, train, measures):
    if not self.store_detections_after_each_epoch or train:
      return
    if self._detection_file is None:
      tf.gfile.MakeDirs("forwarded/" + self.model)
      det_file_path = "forwarded/" + self.model + "/det_" + str(epoch + 1) + ".json"
      self._detection_file = open(det_file_path, "w")
      first = True
    else:
      first = False
    assert Constants.DETECTION_AP in measures
    det_boxes, det_scores, det_classes, num_detections, gt_boxes, gt_classes, gt_ids, n_classes, tags = \
        measures[Constants.DETECTION_AP]
    for bboxes_img, scores_img, classes_img, n_detections_img, tag in zip(det_boxes, det_scores, det_classes,
                                                                          num_detections, tags):
      #tag = tag.split("/")[-1].replace(".png", "")
      for bbox, score, class_ in zip(bboxes_img[:n_detections_img], scores_img[:n_detections_img],
                                     classes_img[:n_detections_img]):
        if first:
          first = False
          self._detection_file.write("[\n")
        else:
          self._detection_file.write(",\n")
        self._detection_file.write("""{{"image_id": "{}", "category_id": {}, "bbox": [{}, {}, {}, {}], \
"score": {}}}""".format(tag, class_ + 1, bbox[2], bbox[0], bbox[3] - bbox[2], bbox[1] - bbox[0], score))

  def _maybe_finalize_detections(self, train):
    if not self.store_detections_after_each_epoch or train:
      return
    assert self._detection_file is not None
    self._detection_file.write("]")
    self._detection_file.close()
    self._detection_file = None
