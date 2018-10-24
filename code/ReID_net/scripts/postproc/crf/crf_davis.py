#!/usr/bin/env python

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.ndimage import imread
from scipy.misc import imsave
import pickle
import numpy
import glob
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys

imgs_path = "/work/mahadevan/data/DAVIS/JPEGImages/480p/"
annots_path = "/work/mahadevan/data/DAVIS/Annotations/480p/"
preds_path_prefix = "/home/mahadevan/vision/savitar/forwarded/"


def convert_path(inp):
  sp = inp.split("/")
  fwd_idx = sp.index("forwarded")

  seq = sp[fwd_idx + 3]
  fn = sp[-1]
  im_path = imgs_path + seq + "/" + fn.replace(".pickle", ".jpg")
  gt_path = annots_path + seq + "/" + fn.replace(".pickle", ".png")

  sp[fwd_idx + 1] += "_crf"
  sp[-1] = sp[-1].replace(".pickle", ".png")
  out_path = "/".join(sp)
  return im_path, gt_path, out_path


def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def apply_crf(im, pred):
  im = numpy.ascontiguousarray(im)
  pred = numpy.ascontiguousarray(pred.swapaxes(0, 2).swapaxes(1, 2))

  d = dcrf.DenseCRF2D(854, 480, 2)  # width, height, nlabels
  unaries = unary_from_softmax(pred, scale=1.0)
  d.setUnaryEnergy(unaries)

  #print im.shape
  # print annot.shape
  #print pred.shape

  d.addPairwiseGaussian(sxy=0.220880737269, compat=1.24845093352)
  d.addPairwiseBilateral(sxy=22.3761305044, srgb=7.70254062277, rgbim=im, compat=1.40326787165)
  processed = d.inference(12)
  res = numpy.argmax(processed, axis=0).reshape(480, 854)

  return res


def do_seq(seq, model, save=True):
  preds_path = preds_path_prefix + model + "/valid/"
  files = sorted(glob.glob(preds_path + seq + "/*.pickle"))
  ious = []
  for f in files:
    pred_path = f
    im_path, gt_path, out_path = convert_path(f)
    pred = pickle.load(open(pred_path))
    im = imread(im_path)
    res = apply_crf(im, pred).astype("uint8") * 255
    # before = numpy.argmax(pred, axis=2)
    if save:
      dir_ = "/".join(out_path.split("/")[:-1])
      mkdir_p(dir_)
      imsave(out_path, res)

    #compute iou as well
    groundtruth = imread(gt_path)
    I = numpy.logical_and(res == 255, groundtruth == 255).sum()
    U = numpy.logical_or(res == 255, groundtruth == 255).sum()
    IOU = float(I) / U
    ious.append(IOU)

    print(out_path, "IOU", IOU)

    # plt.imshow(before)
    # plt.figure()
    # plt.imshow(res)
    # plt.show()
  return numpy.mean(ious[1:-1])


def main():
  #seqs = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl",
  #        "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf", "libby", "motocross-jump",
  #        "paragliding-launch", "parkour", "scooter-black", "soapbox"]
  seqs = ["dance-twirl"]
  save = True
  assert len(sys.argv) == 2
  model = sys.argv[1]
  #model = "paper_nomask_DAVIS1_oneshot1_4"

  #ious = []
  #for seq in seqs:
  #  iou = do_seq(seq, save=save)
  #  print iou
  #  ious.append(iou)

  ious = Parallel(n_jobs=20)(delayed(do_seq)(seq, model, save=save) for seq in seqs)

  print(ious)
  print(numpy.mean(ious))


if __name__ == "__main__":
  main()
