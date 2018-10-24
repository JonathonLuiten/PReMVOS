#!/usr/bin/env python
import glob
import sys
import numpy
from scipy.ndimage import imread

YOUTUBE_PATH = "/data/corpora/youtube-objects/youtube_masks/"
FORWARDED_PATH = "/home/voigtlaender/vision/savitar/forwarded/"


def compute_iou_for_binary_segmentation(y_argmax, target):
  I = numpy.logical_and(y_argmax == 1, target == 1).sum()
  U = numpy.logical_or(y_argmax == 1, target == 1).sum()
  if U == 0:
    IOU = 1.0
  else:
    IOU = float(I) / U
  return IOU


def eval_sequence(gt_folder, recog_folder):
  seq = gt_folder.split("/")[-7] + "_" + gt_folder.split("/")[-5]
  gt_files = sorted(glob.glob(gt_folder + "/*.jpg"))

  gt_files = gt_files[1:]
  recog_folder_seq = recog_folder + seq + "/"
  
  #for full dataset
  recog_files = []
  for gt_file in gt_files:
    idx = int(gt_file.split("/")[-1].replace(".jpg", ""))
    ending = "frame%04d.png" % idx
    recog_file = recog_folder_seq + ending
    recog_files.append(recog_file)

  ious = []
  for gt_file, recog_file in zip(gt_files, recog_files):
    gt = imread(gt_file) / 255
    recog = imread(recog_file) / 255
    iou = compute_iou_for_binary_segmentation(recog, gt)
    ious.append(iou)
  return numpy.mean(ious)


def main():
  assert len(sys.argv) == 2

  pattern = YOUTUBE_PATH + "*/data/*/*/*/labels/"
  folders = glob.glob(pattern)

  # filter out the 2 sequences, which only have a single annotated frame
  folders = [f for f in folders if "motorbike/data/0007/shots/001" not in f and "cow/data/0019/shots/001" not in f]

  ious = {}
  for folder in folders:
    tag = folder.split("/")[-7]
    recog_folder = FORWARDED_PATH + sys.argv[1] + "/valid/"
    iou = eval_sequence(folder, recog_folder)
    #print iou
    if tag in ious:
      ious[tag].append(iou)
    else:
      ious[tag] = [iou]

  class_ious = []
  for k, v in list(ious.items()):
    print(k, numpy.mean(v))
    class_ious.append(numpy.mean(v))
  print("-----")
  print("total", len(class_ious), numpy.mean(class_ious))

if __name__ == "__main__":
  main()
