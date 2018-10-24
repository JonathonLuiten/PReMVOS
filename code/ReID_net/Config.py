from ReID_net.Log import log
import json
from collections import OrderedDict
import tensorflow as tf


class Config(object):
  def __init__(self, filename):
    lines = open(filename).readlines()
    #remove comments (lines starting with #)
    lines = [l if not l.strip().startswith("#") else "\n" for l in lines]
    s = "".join(lines)
    self._entries = json.loads(s, object_pairs_hook=OrderedDict)

    self._initialized = False
    self.dtype = None

  #the full initialization can only be done after the log is initialized as we might need to write to the log
  def initialize(self):
    if self._initialized:
      return
    batch_size = self.int("batch_size")
    gpus = self.int_list("gpus")
    if batch_size % len(gpus) != 0:
      batch_size += len(gpus) - batch_size % len(gpus)
      print("Warning, batch_size not divisible by number of gpus, increasing batch_size to", batch_size, file=log.v1)
      self._entries["batch_size"] = batch_size
    if "fp16" not in self._entries:
      self._entries["fp16"] = False
    self.dtype = tf.float16 if self.bool("fp16") else tf.float32
    self._initialized = True

  def has(self, key):
    return key in self._entries

  def _value(self, key, dtype, default):
    if default is not None:
      assert isinstance(default, dtype)
    if key in self._entries:
      val = self._entries[key]
      if isinstance(val, dtype):
        return val
      else:
        raise TypeError()
    else:
      assert default is not None
      return default

  def _list_value(self, key, dtype, default):
    if default is not None:
      assert isinstance(default, list)
      for x in default:
        assert isinstance(x, dtype)
    if key in self._entries:
      val = self._entries[key]
      assert isinstance(val, list)
      for x in val:
        assert isinstance(x, dtype)
      return val
    else:
      assert default is not None
      return default

  def bool(self, key, default=None):
    return self._value(key, bool, default)

  #def string(self, key, default=None):
  #  return self._value(key, str, default)
  #actually json uses unicode
  def str(self, key, default=None):
    if isinstance(default, str):
      default = str(default)
    return self._value(key, str, default)

  def int(self, key, default=None):
    return self._value(key, int, default)

  def float(self, key, default=None):
    return self._value(key, float, default)

  def dict(self, key, default=None):
    return self._value(key, dict, default)

  def int_key_dict(self, key, default=None):
    if default is not None:
      assert isinstance(default, dict)
      for k in list(default.keys()):
        assert isinstance(k, int)
    dict_str = self.str(key)
    res = eval(dict_str)
    assert isinstance(res, dict)
    for k in list(res.keys()):
      assert isinstance(k, int)
    return res

  def int_list(self, key, default=None):
    return self._list_value(key, int, default)

  def float_list(self, key, default=None):
    return self._list_value(key, float, default)

  def unicode_list(self, key, default=None):
    return self._list_value(key, str, default)

  def dir(self, key, default=None):
    p = self.str(key, default)
    if p[-1] != "/":
      return p + "/"
    else:
      return p
