#!/usr/bin/env python
import sys
import os

from ReID_net.Engine import Engine
from ReID_net.Config import Config
from ReID_net.Log import log
import tensorflow as tf


def init_log(config):
  log_dir = config.dir("log_dir", "logs")
  model = config.str("model")
  filename = log_dir + model + ".log"
  verbosity = config.int("log_verbosity", 3)
  log.initialize([filename], [verbosity], [])


def main(_):
  assert len(sys.argv) == 2, "usage: main.py <config>"
  config_path = sys.argv[1]
  assert os.path.exists(config_path), config_path
  try:
    config = Config(config_path)
  except ValueError as e:
    print("Malformed config file:", e)
    return -1
  init_log(config)
  config.initialize()
  #dump the config into the log
  print(open(config_path).read())
  engine = Engine(config)
  engine.run()

if __name__ == '__main__':
  #for profiling. Note however that this will not be useful for the execution of the tensorflow graph,
  #only for stuff like initialization including creation of the graph, loading of weights, etc.
  #import cProfile
  #cProfile.run("tf.app.run(main)", sort="tottime")

  tf.app.run(main)
