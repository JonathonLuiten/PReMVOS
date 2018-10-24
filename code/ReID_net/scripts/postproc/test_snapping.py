#!/usr/bin/env python

from scipy.ndimage import imread
from scipy.misc import imsave
from scipy.io import loadmat
import numpy
import glob
import os
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

imgs_path = "/work/voigtlaender/data/DAVIS/JPEGImages/480p/"
#annots_path = "/work/voigtlaender/data/DAVIS/Annotations/480p/blackswan/00000.png"
preds_path = "/work/voigtlaender/data/training/2016-01-13-tf-test/forwarded/wide_oneshot2/valid/"
superpixels_path = "/work/voigtlaender/data/COB-DAVIS-all-proposals/"


def convert_path(inp):
  sp = inp.split("/")
  fwd_idx = sp.index("forwarded")

  seq = sp[fwd_idx + 3]
  fn = sp[-1]

  im_path = imgs_path + seq + "/" + fn.replace(".pickle", ".jpg")

  superpixel_path = superpixels_path + seq + "/" + fn.replace(".pickle", ".mat")

  sp[fwd_idx + 1] += "_snapped"
  sp[-1] = sp[-1].replace(".pickle", ".png")
  out_path = "/".join(sp)
  return im_path, superpixel_path, out_path


def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def extract_superpixels(raw):
  #convert to 0-indexing
  raw -= 1
  n_pixels = raw.max()
  pixels = [raw == idx for idx in range(n_pixels)]
  return pixels


def apply_snapping(superpixels, pred):
  extracted = extract_superpixels(superpixels)
  res = numpy.zeros(pred.shape[:2])
  for pix in extracted:
    score = pred[pix, 1].mean()
    res[pix] = score > 0.5
  return res


def do_seq(seq):
  files = sorted(glob.glob(preds_path + seq + "/*.pickle"))
  for f in files:
    pred_path = f
    im_path, superpixel_path, out_path = convert_path(f)
    im = imread(im_path)
    pred = pickle.load(open(pred_path))
    superpixels = loadmat(superpixel_path)["superpixels"]
    res = apply_snapping(superpixels, pred).astype("uint8") * 255
    # before = numpy.argmax(pred, axis=2)
    dir_ = "/".join(out_path.split("/")[:-1])
    mkdir_p(dir_)
    imsave(out_path, res)
    print(out_path)

    #TODO: compute iou as well

    # plt.imshow(before)
    # plt.figure()
    # plt.imshow(res)
    # plt.show()


def main():
  seqs = ["blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl",
          "dog", "drift-chicane", "drift-straight", "goat", "horsejump-high", "kite-surf", "libby", "motocross-jump",
          "paragliding-launch", "parkour", "scooter-black", "soapbox"]
  #for seq in seqs:
  #  do_seq(seq)
  Parallel(n_jobs=10)(delayed(do_seq)(seq) for seq in seqs)


if __name__ == "__main__":
  main()


#params:
#scale: tune by hand
#sxy: leave at 3, biliteral usually higher than gaussian
#compat: weights, maybe we can reduce both to one param, if only the relative weights matter, can also be matrix/array
#5 (now 40 in d.inference(40)) : #iterations, better put more
#srgb: how far to look, like deviation

#tune by hyperopt: sxy (only for bilateral), other at 3, srgb, compat (both or reduce to 1)
