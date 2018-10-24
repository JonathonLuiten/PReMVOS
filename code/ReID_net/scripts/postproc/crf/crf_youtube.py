#!/usr/bin/env python

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.ndimage import imread
from scipy.misc import imsave, imresize
import pickle
import numpy
import glob
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys

#imgs_path = "/work/voigtlaender/data/DAVIS/JPEGImages/480p/"
imgs_path = "/data/corpora/youtube-objects/youtube_masks_full/"
preds_path_prefix = "/fastwork/voigtlaender/mywork/data/training/2016-01-13-tf-test/forwarded/"


def convert_path(inp):
  sp = inp.split("/")
  fwd_idx = sp.index("forwarded")

  seq = sp[fwd_idx + 3]
  fn = sp[-1]

  mainseq = seq.split("_")[0]
  subseq = seq.split("_")[1]
  im_path = imgs_path + mainseq + "/data/" + subseq + "/shots/001/images/" + \
      fn.replace(".pickle", ".jpg")
  #.replace("frame", "")

  sp[fwd_idx + 1] += "_crf"
  sp[-1] = sp[-1].replace(".pickle", ".png")
  out_path = "/".join(sp)
  return im_path, out_path


def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def apply_crf(im, pred):
  im = numpy.ascontiguousarray(im)
  if im.shape[:2] != pred.shape[:2]:
    im = imresize(im, pred.shape[:2])

  pred = numpy.ascontiguousarray(pred.swapaxes(0, 2).swapaxes(1, 2))

  d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], 2)  # width, height, nlabels
  unaries = unary_from_softmax(pred, scale=1.0)
  d.setUnaryEnergy(unaries)

  d.addPairwiseGaussian(sxy=0.220880737269, compat=1.24845093352)
  d.addPairwiseBilateral(sxy=22.3761305044, srgb=7.70254062277, rgbim=im, compat=1.40326787165)
  processed = d.inference(12)
  res = numpy.argmax(processed, axis=0).reshape(im.shape[:2])

  return res


def do_seq(seq, model, save=True):
  preds_path = preds_path_prefix + model + "/valid/"
  files = sorted(glob.glob(preds_path + seq + "/*.pickle"))
  for f in files:
    pred_path = f
    im_path, out_path = convert_path(f)
    pred = pickle.load(open(pred_path))
    im = imread(im_path)
    res = apply_crf(im, pred).astype("uint8") * 255
    # before = numpy.argmax(pred, axis=2)
    if save:
      dir_ = "/".join(out_path.split("/")[:-1])
      mkdir_p(dir_)
      imsave(out_path, res)

    print(out_path)


def main():
  seqs = ["aeroplane_0001", "aeroplane_0002", "aeroplane_0010", "aeroplane_0011", "aeroplane_0012", "aeroplane_0013",
          "bird_0001", "bird_0007", "bird_0010", "bird_0011", "bird_0012", "bird_0014", "boat_0001", "boat_0003",
          "boat_0004", "boat_0005", "boat_0006", "boat_0007", "boat_0008", "boat_0009", "boat_0010", "boat_0011",
          "boat_0012", "boat_0014", "boat_0015", "boat_0016", "boat_0017", "car_0001", "car_0002", "car_0003",
          "car_0004", "car_0005", "car_0008", "car_0009", "cat_0001", "cat_0002", "cat_0003", "cat_0004", "cat_0006",
          "cat_0008", "cat_0010", "cat_0011", "cat_0012", "cat_0013", "cat_0014", "cat_0015", "cat_0016", "cat_0017",
          "cat_0018", "cat_0020", "cow_0001", "cow_0002", "cow_0003", "cow_0004", "cow_0005", "cow_0006", "cow_0007",
          "cow_0008", "cow_0009", "cow_0010", "cow_0011", "cow_0012", "cow_0013", "cow_0014", "cow_0015", "cow_0016",
          "cow_0017", "cow_0018", "cow_0022", "dog_0001", "dog_0003", "dog_0005", "dog_0006", "dog_0007", "dog_0008",
          "dog_0009", "dog_0010", "dog_0012", "dog_0013", "dog_0014", "dog_0016", "dog_0018", "dog_0020", "dog_0021",
          "dog_0022", "dog_0023", "dog_0025", "dog_0026", "dog_0027", "dog_0028", "dog_0030", "dog_0031", "dog_0032",
          "dog_0034", "dog_0035", "dog_0036", "horse_0001", "horse_0009", "horse_0010", "horse_0011", "horse_0012",
          "horse_0014", "horse_0018", "horse_0020", "horse_0021", "horse_0022", "horse_0024", "horse_0025",
          "horse_0026", "horse_0029", "motorbike_0001", "motorbike_0002", "motorbike_0003", "motorbike_0006",
          "motorbike_0009", "motorbike_0011", "motorbike_0012", "motorbike_0013", "motorbike_0014", "train_0001",
          "train_0003", "train_0008", "train_0024", "train_0025"]

  save = True
  assert len(sys.argv) == 2
  model = sys.argv[1]

  #for seq in seqs:
  #  do_seq(seq, model, save=save)

  Parallel(n_jobs=20)(delayed(do_seq)(seq, model, save=save) for seq in seqs)


if __name__ == "__main__":
  main()
