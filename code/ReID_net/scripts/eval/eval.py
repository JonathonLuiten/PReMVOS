#!/usr/bin/env python

from scipy.ndimage import imread
import pickle
import numpy
import glob
import os
from joblib import Parallel, delayed
import sys

from ReID_net.datasets.DAVIS.DAVIS_iterative import get_bounding_box

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
  return im_path, gt_path


def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def do_seq(seq, model):
  preds_path = preds_path_prefix + model + "/valid/"
  files = sorted(glob.glob(preds_path + seq + "/*.pickle"))
  ious = []
  for f in files:
    pred_path = f
    im_path, gt_path = convert_path(f)
    pred = pickle.load(open(pred_path))
    res = numpy.argmax(pred, axis=2) * 255

    #compute iou as well
    groundtruth = imread(gt_path)
    bbox = get_bounding_box(groundtruth, 255)
    res = numpy.logical_and(res / 255, bbox) * 255
    I = numpy.logical_and(res == 255, groundtruth == 255).sum()
    U = numpy.logical_or(res == 255, groundtruth == 255).sum()
    IOU = float(I) / U
    ious.append(IOU)

    print(im_path, "IOU", IOU)

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
  assert len(sys.argv) == 2
  model = sys.argv[1]

  #ious = []
  #for seq in seqs:
  #  iou = do_seq(seq, model)
  #  print iou
  #  ious.append(iou)

  ious = Parallel(n_jobs=20)(delayed(do_seq)(seq, model) for seq in seqs)

  print(ious)
  print(numpy.mean(ious))


if __name__ == "__main__":
  main()
