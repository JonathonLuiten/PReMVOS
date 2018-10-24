import inspect
import sys
import tensorflow as tf

import ReID_net.Constants as Constants
import ReID_net.Measures as Measures
from . import NetworkLayers
from . import NetworkOutputLayers
from . import NetworkSiameseLayers
from ReID_net.Log import log
from .Util_Network import TowerSetup


def get_layer_class(layer_class):
  layers_modules = [NetworkLayers, NetworkOutputLayers, NetworkSiameseLayers]
  class_ = None
  for layer_module in layers_modules:
    if hasattr(layer_module, layer_class):
      assert class_ is None, "class found im multiple modules"
      class_ = getattr(layer_module, layer_class)
  assert class_ is not None, ("Unknown layer class", layer_class)
  return class_


class Network(object):
  def build_tower(self, network_def, x_image, y_ref, tags, void_label, n_classes, tower_setup):
    use_dropout = not tower_setup.is_training
    gpu_str = "/gpu:" + str(tower_setup.gpu)
    if tower_setup.is_main_train_tower:
      print("inputs:", [x_image.get_shape().as_list()])
    with tf.device(gpu_str), tf.name_scope("tower_gpu_" + str(tower_setup.gpu)):
      output_layer = None
      layers = {}
      for name, layer_def in list(network_def.items()):
        layer_def = layer_def.copy()
        layer_class = layer_def["class"]
        if layer_class == "GraphSection":
          if self.use_partialflow:
            if self.current_graph_section is not None:
              self.current_graph_section.__exit__(None, None, None)
            self.current_graph_section = self.graph_section_manager.new_section()
            self.graph_sections.append(self.current_graph_section)
            self.current_graph_section.__enter__()
          # else:
          #  print >> log.v1, "warning, GraphSection defined, but use_partialflow is False. Ignoring sections"
          continue
        if layer_class == "SiameseMerge" or layer_class == "ExpandedSiameseMerge" or \
                layer_class == "CompleteSiameseMerge":
          merge_type = self.config.str("merge_type", "")
          layer_def["merge_type"] = merge_type
        if layer_class == "Clustering" or layer_class == "FullyConnectedWithTripletLoss":
          original_labels = self.inputs_tensors_dict["original_labels"]
          layer_def["original_labels"] = original_labels
        use_image_summaries = self.config.bool("use_image_summaries", False)
        if layer_class == "FullyConnectedWithTripletLoss" and use_image_summaries:
          imgs_raw = self.inputs_tensors_dict["imgs_raw"]
          layer_def["imgs_raw"] = imgs_raw
        del layer_def["class"]
        class_ = get_layer_class(layer_class)
        spec = inspect.getargspec(class_.__init__)
        args = spec[0]

        if "from" in layer_def:
          inputs = sum([layers[x].outputs for x in layer_def["from"]], [])
          del layer_def["from"]
        else:
          inputs = [x_image]
        #option to get the original input images, can e.g. be used for summaries on a higher layer
        if "original_inputs" in args:
          layer_def["original_inputs"] = [x_image]
        if "tags" in args:
          layer_def["tags"] = tags
        if "concat" in layer_def:
          concat = sum([layers[x].outputs for x in layer_def["concat"]], [])
          layer_def["concat"] = concat
        if "alternative_labels" in layer_def:
          # if tower_setup.is_training:
          layer_def["targets"] = sum([layers[x].out_labels for x in layer_def["alternative_labels"]])
          layer_def["n_classes"] = 2
          # else:
          #   layer_def["targets"] = y_ref
          #   layer_def["n_classes"] = 2
          del layer_def["alternative_labels"]
        elif class_.output_layer:
          layer_def["targets"] = y_ref
          layer_def["n_classes"] = n_classes
          if "void_label" in args:
            layer_def["void_label"] = void_label
          if "imgs_raw" in args:
            layer_def["imgs_raw"] = x_image
        else:
          if "targets" in args:
            layer_def["targets"] = y_ref
          if "n_classes" in args:
            layer_def["n_classes"] = n_classes
        layer_def["name"] = name
        layer_def["inputs"] = inputs
        if "dropout" in args and not use_dropout:
          layer_def["dropout"] = 0.0
        if "tower_setup" in args:
          layer_def["tower_setup"] = tower_setup

        # check if all args are specified
        defaults = spec[3]
        if defaults is None:
          defaults = []
        n_non_default_args = len(args) - len(defaults)
        non_default_args = args[1:n_non_default_args]  # without self
        for arg in non_default_args:
          assert arg in layer_def, (name, arg)

        layer = class_(**layer_def)

        if "stride" in layer_def:
          inp_size = x_image.get_shape().as_list()[-2]
          out_size = inputs[-1].get_shape().as_list()[-2]
          # note that these sizes are not always available, so we we cannot just set it,
          # but at least we can sometimes check it
          if inp_size is not None and out_size is not None and round(float(inp_size) / out_size) != layer_def["stride"]:
            print("error, wrong stride specified in config!")
            sys.exit(1)

        if tower_setup.is_main_train_tower:
          print(name, "shape:")
          for l in layer.outputs:
            if isinstance(l, tuple) or isinstance(l, list):
              for v in l:
                if v is None:
                  print("None")
                else:
                  print(v.get_shape().as_list())
            else:
              print(l.get_shape().as_list())
        layers[name] = layer
        if class_.output_layer:
          assert output_layer is None, "Currently only 1 output layer is supported"
          output_layer = layer
      assert output_layer is not None, "No output layer in network"

      if isinstance(y_ref, tuple) or isinstance(y_ref, list):
        n = tf.shape(y_ref[0])[0]
      else:
        n = tf.shape(y_ref)[0]
      assert len(output_layer.outputs) == 1, len(output_layer.outputs)
      loss, measures, y_softmax = output_layer.loss, output_layer.measures, output_layer.outputs[0]
      regularizers_tower = []
      update_ops_tower = []
      for l in list(layers.values()):
        self.summaries += l.summaries
        regularizers_tower += l.regularizers
        update_ops_tower += l.update_ops
      n_params = sum([l.n_params for l in list(layers.values())])
      return loss, measures, y_softmax, n, n_params, regularizers_tower, update_ops_tower, layers

  def build_network(self, config, x_image, y_ref, tags, void_label, n_classes, is_training, freeze_batchnorm,
                    use_weight_summaries):
    gpus = config.int_list("gpus")
    # only use one gpu for eval
    if not is_training:
      gpus = gpus[:1]
    if self.use_partialflow:
      assert len(gpus) == 1, len(gpus)  # partialflow does not work with multigpu
    network_def = config.dict("network")
    batch_size_tower = self.batch_size / len(gpus)
    assert batch_size_tower * len(gpus) == self.batch_size, (batch_size_tower, len(gpus), self.batch_size)
    loss_summed = measures_accumulated = y_softmax_total = n_total = n_params = None
    tower_losses = []
    tower_regularizers = []
    update_ops = []
    tower_setups = []
    tower_layers = []
    first = True
    if x_image.get_shape().as_list()[0] is not None:
      if self.chunk_size != -1:
        assert x_image.get_shape().as_list()[0] == self.batch_size * self.chunk_size, \
          "dataset produced inputs with wrong shape"
      else:
        assert x_image.get_shape().as_list()[0] == self.batch_size, "dataset produced inputs with wrong batch size"
    for idx, gpu in enumerate(gpus):
      original_sizes = self.inputs_tensors_dict.get(Constants.ORIGINAL_SIZES, None)
      resized_sizes = self.inputs_tensors_dict.get(Constants.RESIZED_SIZES, None)
      if len(gpus) == 1:
        x_image_tower = x_image
        y_ref_tower = y_ref
        tags_tower = tags
        variable_device = "/gpu:0"
      else:
        stride = batch_size_tower * (1 if self.chunk_size == -1 else self.chunk_size)
        x_image_tower = x_image[idx * stride:(idx + 1) * stride]
        tags_tower = tags[idx * stride:(idx + 1) * stride]
        if original_sizes is not None:
          original_sizes = original_sizes[idx * stride:(idx + 1) * stride]
        if resized_sizes is not None:
          resized_sizes = resized_sizes[idx * stride:(idx + 1) * stride]
        if isinstance(y_ref, tuple):
          y_ref_tower = tuple(v[idx * stride: (idx + 1) * stride] for v in y_ref)
        else:
          y_ref_tower = y_ref[idx * stride: (idx + 1) * stride]
        variable_device = "/cpu:0"

      is_main_train_tower = is_training and first
      tower_setup = TowerSetup(dtype=config.dtype, gpu=gpu, is_main_train_tower=is_main_train_tower,
                               is_training=is_training, freeze_batchnorm=freeze_batchnorm,
                               variable_device=variable_device, use_update_ops_collection=self.use_partialflow,
                               batch_size=batch_size_tower, original_sizes=original_sizes, resized_sizes=resized_sizes,
                               use_weight_summaries=is_main_train_tower and use_weight_summaries)
      tower_setups.append(tower_setup)

      with tf.variable_scope(tf.get_variable_scope(), reuse=True if not first else None):
        loss, measures, y_softmax, n, n_params_tower, regularizers, update_ops_tower, layers = self.build_tower(
          network_def, x_image_tower, y_ref_tower, tags_tower, void_label, n_classes, tower_setup)

      tower_layers.append(layers)
      tower_losses.append(loss / tf.cast(n, tower_setup.dtype))
      tower_regularizers.append(regularizers)
      if first:
        loss_summed = loss
        measures_accumulated = measures
        y_softmax_total = [y_softmax]
        n_total = n
        update_ops = update_ops_tower
        first = False
        n_params = n_params_tower
      else:
        loss_summed += loss
        measures_accumulated = Measures.calc_measures_sum(measures_accumulated, measures)
        y_softmax_total.append(y_softmax)
        n_total += n
        update_ops += update_ops_tower
        assert n_params_tower == n_params
    if len(gpus) == 1:
      y_softmax_total = y_softmax_total[0]
    else:
      if isinstance(y_softmax_total[0], tuple):
        y_softmax_out = []
        for n in range(len(y_softmax_total[0])):
          if y_softmax_total[0][n] is None:
            #TODO: or just leave it out?
            y_softmax_out.append(None)
          else:
            v = tf.concat(axis=0, values=[y[n] for y in y_softmax_total])
            y_softmax_out.append(v)
        y_softmax_total = tuple(y_softmax_out)
      else:
        y_softmax_total = tf.concat(axis=0, values=y_softmax_total, name='y_softmax_total')
    if self.current_graph_section is not None:
      self.current_graph_section.__exit__(None, None, None)
    return tower_losses, tower_regularizers, loss_summed, y_softmax_total, measures_accumulated, n_total, n_params, \
        update_ops, tower_setups, tower_layers

  def __init__(self, config, dataset, global_step, training, use_partialflow=False, freeze_batchnorm=False,
               name=""):
    with tf.name_scope(name):
      self.config = config
      self.use_partialflow = use_partialflow
      use_weight_summaries = config.bool("use_weight_summaries", False)
      if use_partialflow:
        import partialflow
        self.graph_section_manager = partialflow.GraphSectionManager(verbose=False)
        self.graph_sections = []
      else:
        self.graph_section_manager = None
      self.current_graph_section = None
      if training:
        self.batch_size = config.int("batch_size")
        self.chunk_size = config.int("chunk_size", -1)
      else:
        assert freeze_batchnorm
        self.chunk_size = config.int("eval_chunk_size", -1)
        if self.chunk_size == -1:
          self.chunk_size = config.int("chunk_size", -1)

        do_multi_sample_testing = config.int("n_test_samples", -1) != -1
        if do_multi_sample_testing:
          self.batch_size = 1
        else:
          self.batch_size = config.int("batch_size_eval", -1)
          if self.batch_size == -1:
            self.batch_size = config.int("batch_size")
      n_classes = dataset.num_classes()
      if config.bool("adjustable_output_layer", False):
        n_classes = None
      self.global_step = global_step
      self.inputs_tensors_dict = dataset.create_input_tensors_dict(self.batch_size)
      # inputs and labels are not optional
      self.inputs = self.inputs_tensors_dict["inputs"]
      labels = self.inputs_tensors_dict["labels"]
      self.raw_labels = self.inputs_tensors_dict.get("raw_labels", None)
      self.index_imgs = self.inputs_tensors_dict.get("index_imgs", None)
      self.tags = self.inputs_tensors_dict.get("tags")
      self.img_ids = self.inputs_tensors_dict.get(Constants.IMG_IDS, None)

      void_label = dataset.void_label()
      # important: first inputs_and_labels (which creates summaries) and then access summaries
      self.summaries = []
      self.summaries += dataset.summaries
      self.losses, self.regularizers, self.loss_summed, self.y_softmax, self.measures_accumulated, self.n_imgs, \
          self.n_params, self.update_ops, self.tower_setups, self.tower_layers = self.build_network(
            config, self.inputs, labels, self.tags, void_label, n_classes, training, freeze_batchnorm,
        use_weight_summaries)

      # e.g. used for resizing
      if self.raw_labels is not None and not isinstance(self.y_softmax, tuple):
        self.ys_resized = tf.image.resize_images(self.y_softmax, tf.shape(self.raw_labels)[1:3])
        self.ys_argmax = tf.argmax(self.ys_resized, 3)

  def get_output_layer(self):
    layers = self.tower_layers[0]
    output_layers = [l for l in list(layers.values()) if l.output_layer]
    assert len(output_layers) == 1
    return output_layers[0]
