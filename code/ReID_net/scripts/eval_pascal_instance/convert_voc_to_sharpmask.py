import pickle
import glob
import os
import numpy as np
from scipy import misc
import scipy.io
from ReID_net.datasets.Util import Util as dataUtil

VOC_PATH = "/work/" + dataUtil.username() + "/data/PascalVOC/benchmark_RELEASE/dataset/"
OUT_FN = "list.txt"
VOID_LABEL = 255

def main():
  files = glob.glob( VOC_PATH + "dets/*.pickle")
  out_file = open(OUT_FN, 'w')
  for file in files:
    filename_without_ext = file.split('/')[-1].split('.')[-2]
    boxes = pickle.load( open(file) )

    for box in boxes:
      if box.shape[0] != 0:
        [x_min, y_min, x_max, y_max] = box[0][:4].astype(int)
        data = VOC_PATH + "JPEGImages/" + filename_without_ext + ' ' + repr(x_min) + ' ' + repr(y_min) \
               + ' ' + repr(x_max - x_min) + ' ' + repr(y_max - y_min) + "\n"
        out_file.write(data)


def gt_to_sharpmask():
  out_file = open(OUT_FN, 'w')
  val_images = "datasets/PascalVOC/val.txt"
  for im in open(val_images):
    instance_segm=None
    file_name_without_ext = im.split('/')[-1].split('.')[-2]
    inst_path = VOC_PATH + "inst/" + file_name_without_ext + ".mat"

    if os.path.exists(inst_path):
      instance_segm = scipy.io.loadmat(inst_path)['GTinst']['Segmentation'][0][0]
    else:
      inst_path = VOC_PATH + "/SegmentationObject/" + file_name_without_ext + ".png"
      if os.path.exists(inst_path):
        instance_segm = misc.imread(inst_path)
      else:
        print("File: " + im + " does not have any instance annotations.")

    if instance_segm is not None:
      inst_labels = np.unique(instance_segm)
      inst_labels = np.setdiff1d(inst_labels,
                                 [0, VOID_LABEL])
      for inst in inst_labels:
        # Create bounding box from segmentation mask.
        rows = np.where(instance_segm == inst)[0]
        cols = np.where(instance_segm == inst)[1]
        rmin = rows.min()
        rmax = rows.max()
        cmin = cols.min()
        cmax = cols.max()
        area = (rmax - rmin) * (cmax - cmin)
        [x_min, y_min, x_max, y_max] = [cmin, rmin, cmax, rmax]
        if area > 200:
          data = VOC_PATH + "JPEGImages/" + file_name_without_ext + ' ' + repr(x_min) + ' ' + repr(y_min) \
            + ' ' + repr(x_max - x_min) + ' ' + repr(y_max - y_min) + "\n"
          out_file.write(data)

if __name__ == '__main__':
    gt_to_sharpmask()
