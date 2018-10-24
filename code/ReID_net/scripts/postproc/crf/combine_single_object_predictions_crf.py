#!/usr/bin/env python
import sys
import glob
import os
import os.path
import pickle
import numpy
from scipy.ndimage import imread
from PIL import Image
from joblib import Parallel, delayed
from ReID_net.datasets.Util.pascal_colormap import save_with_pascal_colormap
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

BASE_DIR = "/home/voigtlaender/vision/savitar/forwarded/"
DAVIS2017_DIR = "/fastwork/voigtlaender/mywork/data/DAVIS2017/"
#SPLIT = "val"
SPLIT = "test-dev"
PARALLEL = True
SAVE = True


def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def run_multiclass_crf(seq, fn, posteriors, softmax_scale, sxy1, compat1, sxy2, compat2, srgb):
  im_fn = DAVIS2017_DIR + "JPEGImages/480p/" + seq + "/" + fn.replace(".pickle", ".jpg")
  im = imread(im_fn)
  nlabels = posteriors.shape[-1]

  im = numpy.ascontiguousarray(im)
  pred = numpy.ascontiguousarray(posteriors.swapaxes(0, 2).swapaxes(1, 2))

  d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], nlabels)  # width, height, nlabels
  unaries = unary_from_softmax(pred, scale=softmax_scale)
  d.setUnaryEnergy(unaries)

  d.addPairwiseGaussian(sxy=sxy1, compat=compat1)
  d.addPairwiseBilateral(sxy=sxy2, srgb=srgb, rgbim=im, compat=compat2)
  processed = d.inference(12)
  res = numpy.argmax(processed, axis=0).reshape(im.shape[:2])
  return res


def calculate_mean_iou(seq, fn, result, n_objects):
  if "test" in SPLIT:
    return 0.0
  mask_fn = DAVIS2017_DIR + "Annotations/480p/" + seq + "/" + fn.replace(".pickle", ".png")
  groundtruth_mask = numpy.array(Image.open(mask_fn))

  iou_sum = 0.0
  for n in range(n_objects):
    I = numpy.logical_and(result == n + 1, groundtruth_mask == n + 1).sum()
    U = numpy.logical_or(result == n + 1, groundtruth_mask == n + 1).sum()
    if U == 0:
      iou = 1.0
    else:
      iou = float(I) / U
    iou_sum += iou
  return iou_sum / n_objects


def do_seq(model, seq, save, softmax_scale, sxy1, compat1, sxy2, compat2, srgb):
  pattern = BASE_DIR + model + "/valid/" + seq + "/*/"
  out_folder = BASE_DIR + model + "_mergedcrf/valid/" + seq + "/"
  if save:
    mkdir_p(out_folder)
  object_folders = sorted(glob.glob(pattern))
  n_objects = len(object_folders)
  filenames = sorted(map(os.path.basename, glob.glob(object_folders[0] + "/*.pickle")))
  ious = []
  for fn in filenames:
    posteriors = [pickle.load(open(obj_folder + "/" + fn)) for obj_folder in object_folders]
    background = numpy.stack([x[..., 0] for x in posteriors], axis=2).min(axis=2, keepdims=True)
    rest = numpy.stack([x[..., 1] for x in posteriors], axis=2)
    rest_rescaled = (1.0 - background) * rest / rest.sum(axis=-1, keepdims=True)
    combined = numpy.concatenate([background, rest_rescaled], axis=2)

    result = run_multiclass_crf(seq, fn, combined, softmax_scale, sxy1, compat1, sxy2, compat2, srgb)
    #result = combined.argmax(axis=-1)

    iou = calculate_mean_iou(seq, fn, result, n_objects)
    ious.append(iou)

    out_fn = out_folder + fn.replace(".pickle", ".png")
    print(out_fn, iou, n_objects)
    if save:
      save_with_pascal_colormap(out_fn, result)
  iou_total = numpy.mean(ious[1:-1])
  print("sequence", seq, "iou", iou_total)
  return iou_total, n_objects


def main():
  assert len(sys.argv) in (2, 8)
  model = sys.argv[1].replace("forwarded/", "")
  if model.endswith("/"):
    model = model[:-1]

  if len(sys.argv) == 2:
    softmax_scale = 1.0
    sxy1 = 0.919403488507
    compat1 = 1.55891804121
    sxy2 = 12.1033212446
    compat2 = 2.78261754779
    srgb = 5.15563483356
  else:
    softmax_scale = float(sys.argv[2])
    sxy1 = float(sys.argv[3])
    compat1 = float(sys.argv[4])
    sxy2 = float(sys.argv[5])
    compat2 = float(sys.argv[6])
    srgb = float(sys.argv[7])

  print("model", model)
  print("softmax_scale", softmax_scale)
  print("sxy1", sxy1)
  print("compat1", compat1)
  print("sxy2", sxy2)
  print("compat2", compat2)
  print("srgb", srgb)

  seqs = list(map(str.strip, open(DAVIS2017_DIR + "ImageSets/2017/" + SPLIT + ".txt").readlines()))

  if PARALLEL:
    results = Parallel(n_jobs=20)(delayed(do_seq)(model, seq, SAVE, softmax_scale, sxy1, compat1, sxy2, compat2, srgb)
                                  for seq in seqs)
    iou_seqs = [r[0] for r in results]
    n_objects_seqs = [r[1] for r in results]
  else:
    iou_seqs = []
    n_objects_seqs = []
    for seq in seqs:
      print(seq)
      iou, n_objects = do_seq(model, seq, SAVE, softmax_scale, sxy1, compat1, sxy2, compat2, srgb)
      iou_seqs.append(iou)
      n_objects_seqs.append(n_objects)
  iou = sum([i * n for i, n in zip(iou_seqs, n_objects_seqs)]) / sum(n_objects_seqs)
  print("total iou", iou)

if __name__ == "__main__":
  main()
