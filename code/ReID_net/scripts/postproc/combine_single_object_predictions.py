#!/usr/bin/env python
import sys
import glob
import os
import os.path
import pickle
import numpy
from joblib import Parallel, delayed
from ReID_net.datasets.Util.pascal_colormap import save_with_pascal_colormap

BASE_DIR = "/home/voigtlaender/vision/savitar/forwarded/"
#PARALLEL = False
PARALLEL = True


def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def do_seq(model, seq):
  pattern = BASE_DIR + model + "/valid/" + seq + "/*/"
  out_folder = BASE_DIR + model + "_merged/valid/" + seq + "/"
  mkdir_p(out_folder)
  object_folders = sorted(glob.glob(pattern))
  filenames = sorted(map(os.path.basename, glob.glob(object_folders[0] + "/*.pickle")))
  for fn in filenames:
    posteriors = [pickle.load(open(obj_folder + "/" + fn)) for obj_folder in object_folders]
    background = numpy.stack([x[..., 0] for x in posteriors], axis=2).min(axis=2, keepdims=True)
    rest = numpy.stack([x[..., 1] for x in posteriors], axis=2)
    combined = numpy.concatenate([background, rest], axis=2)
    result = combined.argmax(axis=2)
    out_fn = out_folder + fn.replace(".pickle", ".png")
    print(out_fn)
    save_with_pascal_colormap(out_fn, result)


def main():
  assert len(sys.argv) == 2
  model = sys.argv[1].replace("forwarded/", "")
  if model.endswith("/"):
    model = model[:-1]
  
  #split = "val"
  split = "test-dev"
  #split = "test-challenge"
  seqs = list(map(str.strip, open("/fastwork/voigtlaender/mywork/data/DAVIS2017/ImageSets/2017/" + split + ".txt").readlines()))
  
  if PARALLEL:
    Parallel(n_jobs=20)(delayed(do_seq)(model, seq) for seq in seqs)
  else:
    for seq in seqs:
      print(seq)
      do_seq(model, seq)

if __name__ == "__main__":
  main()
