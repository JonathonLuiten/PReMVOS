import pickle

import numpy
import tensorflow as tf

from ReID_net.Log import log


# from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def clip_gradients(grads, threshold):
  gradients, variables = list(zip(*grads))
  gradients, norm = tf.clip_by_global_norm(gradients, threshold)
  grads = list(zip(gradients, variables))
  return grads, norm


def load_wider_or_deeper_mxnet_model(model_path, session):
  params = pickle.load(open(model_path))
  vars = tf.global_variables()

  model_name = model_path.split("/")[-1]
  if model_name.startswith("ilsvrc"):
    layer_mapping = {"res0": "res2a", "res1": "res2b1", "res2": "res2b2", "res3": "res3a", "res4": "res3b1",
                     "res5": "res3b2", "res6": "res4a", "res7": "res4b1", "res8": "res4b2", "res9": "res4b3",
                     "res10": "res4b4", "res11": "res4b5", "res12": "res5a", "res13": "res5b1", "res14": "res5b2",
                     "res15": "res6a", "res16": "res7a", "output": "linear1000", "conv0": "conv1a",
                     "collapse": "bn7"}
  elif model_name.startswith("ade"):
    layer_mapping = {"res0": "res2a", "res1": "res2b1", "res2": "res2b2", "res3": "res3a", "res4": "res3b1",
                     "res5": "res3b2", "res6": "res4a", "res7": "res4b1", "res8": "res4b2", "res9": "res4b3",
                     "res10": "res4b4", "res11": "res4b5", "res12": "res5a", "res13": "res5b1", "res14": "res5b2",
                     "res15": "res6a", "res16": "res7a", "output": "linear150", "conv0": "conv1a",
                     "conv1": ["bn7", "conv6a"]}
  elif model_name.startswith("voc"):
    layer_mapping = {"res0": "res2a", "res1": "res2b1", "res2": "res2b2", "res3": "res3a", "res4": "res3b1",
                     "res5": "res3b2", "res6": "res4a", "res7": "res4b1", "res8": "res4b2", "res9": "res4b3",
                     "res10": "res4b4", "res11": "res4b5", "res12": "res5a", "res13": "res5b1", "res14": "res5b2",
                     "res15": "res6a", "res16": "res7a", "output": "linear21", "conv0": "conv1a",
                     "conv1": ["bn7", "conv6a"]}
  else:
    assert False, model_name

  # from str (without :0) to var
  var_dict = {v.name[:-2]: v for v in vars if "Adam" not in v.name and "_power" not in v.name
              and "global_step" not in v.name}

  # from our var name to mxnet var name
  mxnet_dict = create_mxnet_dict(layer_mapping, var_dict)

  for k, v in list(mxnet_dict.items()):
    assert v in params, (k, v)

  # use a placeholder to avoid memory issues
  placeholder = tf.placeholder(tf.float32)
  for idx, (k, v) in enumerate(mxnet_dict.items()):
    print(idx, "/", len(mxnet_dict), "loading", k, file=log.v5)
    val = params[v]
    if val.ndim == 1:
      pass
    elif val.ndim == 2:
      val = numpy.swapaxes(val, 0, 1)
    elif val.ndim == 4:
      val = numpy.moveaxis(val, [0, 1, 2, 3], [3, 2, 0, 1])
    else:
      assert False, val.ndim
    var = var_dict[k]
    if var.get_shape() == val.shape:
      op = tf.assign(var, placeholder)
      session.run([op], feed_dict={placeholder: val})
    elif k.startswith("conv0"):
      print("warning, sizes for", k, "do not match, initializing matching part assuming " \
                                                "the first 3 dimensions are RGB", file=log.v1)
      val_new = session.run(var)
      val_new[..., :3, :] = val
      op = tf.assign(var, placeholder)
      session.run([op], feed_dict={placeholder: val_new})
    else:
      print("skipping", k, "since the shapes do not match:", var.get_shape(), "and", val.shape, file=log.v1)


def create_mxnet_dict(layer_mapping, var_dict):
  mxnet_dict = {}
  for vn in var_dict:
    sp = vn.split("/")
    if sp[0] not in layer_mapping:
      print("warning,", vn, "not found in mxnet model", file=log.v1)
      continue
    layer = layer_mapping[sp[0]]
    if "bn" in sp[1]:
      if isinstance(layer, list):
        layer = layer[0]
      layer = layer.replace("res", "bn")
      if sp[2] == "beta":
        postfix = "_beta"
      elif sp[2] == "gamma":
        postfix = "_gamma"
      elif sp[2] == "mean_ema":
        postfix = "_moving_mean"
      elif sp[2] == "var_ema":
        postfix = "_moving_var"
      else:
        assert False, sp
    else:
      if isinstance(layer, list):
        layer = layer[1]
      postfix = "_weight"

    if "ema" in vn:
      layer = "aux:" + layer
    else:
      layer = "arg:" + layer

    if sp[1] == "W0":
      branch = "_branch1"
    elif sp[1] == "W1":
      branch = "_branch2a"
    elif sp[1] == "W2":
      branch = "_branch2b1"
    elif sp[1] == "W3":
      branch = "_branch2b2"
    elif sp[1] == "W":
      branch = ""
    elif sp[1] == "bn0":
      branch = "_branch2a"
    elif sp[1] == "bn2":
      branch = "_branch2b1"
    elif sp[1] == "bn3":
      branch = "_branch2b2"
    # for collapse
    elif sp[1] == "bn":
      branch = ""
    elif sp[1] == "b":
      branch = ""
      postfix = "_bias"
    else:
      assert False, sp

    mxnet_dict[vn] = layer + branch + postfix
  return mxnet_dict


def debug_is_zero(datum, tensor):
  _ = datum  # Datum metadata is unused in this predicate.
  from tensorflow.python.debug.lib.debug_data import InconvertibleTensorProto

  if isinstance(tensor, InconvertibleTensorProto):
    # Uninitialized tensor doesn't have bad numerical values.
    # Also return False for data types that cannot be represented as numpy
    # arrays.
    return False
  elif (numpy.issubdtype(tensor.dtype, numpy.float) or
        numpy.issubdtype(tensor.dtype, numpy.complex) or
          numpy.issubdtype(tensor.dtype, numpy.integer)):
    return numpy.all(tensor == 0)
  else:
    return False
