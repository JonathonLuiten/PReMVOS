#!/usr/bin/env python

from scipy.ndimage import imread
import pickle
import numpy
import glob
import os
from joblib import Parallel, delayed
import sys
from skimage import measure
from scipy.ndimage.morphology import grey_dilation

imgs_path = "/work/voigtlaender/data/DAVIS/JPEGImages/480p/"
annots_path = "/work/voigtlaender/data/DAVIS/Annotations/480p/"
preds_path_prefix = "/work/voigtlaender/data/training/2016-01-13-tf-test/forwarded/"
objectness_path_prefix = "/work/voigtlaender/data/training/2016-01-13-tf-test/forwarded/paper_DAVIS_objectness2/"


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


def postproc_posteriors_connected_components(posteriors):
  res = numpy.argmax(posteriors, axis=2)
  dilated = grey_dilation(res, size=(100, 100))
  components, n = measure.label(dilated, return_num=True)
  components[res == 0] = 0

  #find largest component
  largest = 0
  largest_count = 0
  for k in range(1, n + 1):
    count = (components == k).sum()
    if count > largest_count:
      largest = k
      largest_count = count

  res = numpy.zeros_like(res)
  if n > 0:
    res[components == largest] = 1

  #import matplotlib.pyplot as plt
  #plt.imshow(res, cmap="spectral")
  #plt.show()
  return res


def postproc_posteriors_objectness(posteriors, pred_path):
  sp = pred_path.split("/")
  postfix = "/".join(sp[-3:])
  objectness_path = objectness_path_prefix + postfix
  objectness = pickle.load(open(objectness_path))

  objectness_scale = 0.3
  #refined_posteriors = objectness_scale * objectness + (1.0 - objectness_scale) * posteriors
  refined_posteriors = (objectness ** 0.5) * posteriors
  return refined_posteriors.argmax(axis=2)


def do_seq(seq, model):
  preds_path = preds_path_prefix + model + "/valid/"
  files = sorted(glob.glob(preds_path + seq + "/*.pickle"))
  ious = []
  for f in files:
    pred_path = f
    im_path, gt_path = convert_path(f)
    pred = pickle.load(open(pred_path))

    #res = numpy.argmax(pred, axis=2)
    #res = postproc_posteriors_connected_components(pred)
    res = postproc_posteriors_objectness(pred, pred_path)
    #res = postproc_posteriors_temporal(pred, lastmask)

    res *= 255

    #compute iou as well
    groundtruth = imread(gt_path)
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
  parallel = True
  seqs = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl",
          "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf", "libby", "motocross-jump",
          "paragliding-launch", "parkour", "scooter-black", "soapbox"]
  assert len(sys.argv) == 2
  model = sys.argv[1]

  if parallel:
    ious = Parallel(n_jobs=20)(delayed(do_seq)(seq, model) for seq in seqs)
  else:
    ious = []
    for seq in seqs:
      iou = do_seq(seq, model)
      print(iou)
      ious.append(iou)

  print(ious)
  print(numpy.mean(ious))


if __name__ == "__main__":
  main()
