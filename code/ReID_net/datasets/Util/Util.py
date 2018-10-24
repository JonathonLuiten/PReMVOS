import itertools as it
import os
import random
from scipy.ndimage import distance_transform_edt

import cv2
import numpy as np
from skimage import color, morphology

from ReID_net.datasets.Util.flo_Reader import read_flo_file
from ReID_net.datasets.Util.python_pfm import readPFM

D = 40
D_MARGIN = 5
# Number of positive clicks to sample
Npos = 5
# Number of negative clicks to sample using strategy 1, 2 and 3 respectively of https://arxiv.org/abs/1603.04042
Nneg1 = 10
Nneg2 = 5
Nneg3 = 10


def unique_list(l):
  res = []
  for x in l:
    if x not in res:
      res.append(x)
  return res


def create_index_image(height, width):
  import tensorflow as tf
  y = tf.range(height)
  x = tf.range(width)
  grid = tf.meshgrid(x, y)
  index_img = tf.stack((grid[1], grid[0]), axis=2)
  return index_img


def smart_shape(x):
  import tensorflow as tf
  shape = x.get_shape().as_list()
  tf_shape = tf.shape(x)
  for i, s in enumerate(shape):
    if s is None:
      shape[i] = tf_shape[i]
  return shape


def read_pfm(fn):
  return readPFM(fn)[0]


def username():
  return os.environ["USER"]


def _postprocess_flow(x, flow_as_angle):
  if flow_as_angle:
    assert False, "not implemented yet"
  else:
    # divide by 20 to get to a more useful range
    x /= 20.0
  return x


def load_flow_from_pfm(fn, flow_as_angle=False):
  # 3rd channel is all zeros
  flow = read_pfm(fn)[:, :, :2]
  flow = _postprocess_flow(flow, flow_as_angle)
  return flow


def load_flow_from_flo(fn, flow_as_angle):
  flow = read_flo_file(fn)
  flow = _postprocess_flow(flow, flow_as_angle)
  return flow


def get_masked_image(img, mask, multiplier=0.6):
  """
  :param img: The image to be masked.
  :param mask: Binary mask to be applied. The object should be represented by 1 and the background by 0
  :param multiplier: Floating point multiplier that decides the colour of the mask.
  :return: Masked image
  """
  img_mask = np.zeros_like(img)
  indices = np.where(mask == 1)
  img_mask[indices[0], indices[1], 1] = 1
  img_mask_hsv = color.rgb2hsv(img_mask)
  img_hsv = color.rgb2hsv(img)
  img_hsv[indices[0], indices[1], 0] = img_mask_hsv[indices[0], indices[1], 0]
  img_hsv[indices[0], indices[1], 1] = img_mask_hsv[indices[0], indices[1], 1] * multiplier

  return color.hsv2rgb(img_hsv)


def get_masked_image_hsv(img_hsv, mask, multiplier=0.6):
  """
  :param img_hsv: The hsv image to be masked.
  :param mask: Binary mask to be applied. The object should be represented by 1 and the background by 0
  :param multiplier: Floating point multiplier that decides the colour of the mask.
  :return: Masked image
  """
  img_mask_hsv = np.zeros_like(img_hsv)
  result_image = np.copy(img_hsv)
  indices = np.where(mask == 1)
  img_mask_hsv[indices[0], indices[1], :] = [0.33333333333333331, 1.0, 0.0039215686274509803]
  result_image[indices[0], indices[1], 0] = img_mask_hsv[indices[0], indices[1], 0]
  result_image[indices[0], indices[1], 1] = img_mask_hsv[indices[0], indices[1], 1] * multiplier

  return color.hsv2rgb(result_image)


def create_distance_transform(img, label, raw_label, strategy, ignore_classes, old_label=None):
  u0, neg_clicks = get_neg_dst_transform(raw_label[:, :, 0], img, 1, strategy, ignore_classes)
  u1, pos_clicks = get_pos_dst_transform(label[:, :, 0], img, 1)
  num_clicks = len(neg_clicks) + len(pos_clicks)

  return u0.astype(np.float32), u1.astype(np.float32), num_clicks


def geo_dist(img, pts):
  # Import these only on demand since pyximport interferes with pycocotools
  import pyximport
  pyximport.install()
  from ReID_net.datasets.Util import sweep

  img = np.copy(img) / 255.0
  #G = nd.gaussian_gradient_magnitude(img, 1.0)
  img = cv2.GaussianBlur(img, (3,3), 1.0)
  #G = cv2.Laplacian(img,cv2.CV_64F)
  sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
  sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
  sobel_abs = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
  sobel_abs = (sobel_abs[:, :, 0] ** 2 + sobel_abs[:, :, 1] ** 2 + sobel_abs[:, :, 2] ** 2) ** (1 / 2.0)

  #G = (G[:, :, 0] ** 2 + G[:, :, 1] ** 2 + G[:, :, 2] ** 2) ** (1 / 2.0)
  # c = 1 + G * 200
  # c = G / np.max(G)
  #c=sobel_abs / 255.0
  c=1+sobel_abs
  # plt.imshow(sobel_abs)
  # plt.colorbar()
  # plt.show()

  dt = np.zeros_like(c)
  dt[:] = 1000
  dt[pts] = 0
  sweeps = [dt, dt[:, ::-1], dt[::-1], dt[::-1, ::-1]]
  costs = [c, c[:, ::-1], c[::-1], c[::-1, ::-1]]
  for i, (a, c) in enumerate(it.cycle(list(zip(sweeps, costs)))):
    # print i,
    if sweep.sweep(a, c) < 1.0 or i >= 40:
      break
  return dt


def get_pos_dst_transform(label_unmodified, img, instance, old_label=None, dt_method="edt"):
  label = np.where(label_unmodified == instance, 1, 0)

  # If an old label is available, then sample positive clicks on the difference between the two.
  if old_label is not None:
    # The difference should be taken only if there is atleast one object pixel in the difference.
    label = np.max(0, label - old_label) if np.any((label - old_label) == 1) else label

  # Leave a margin around the object boundary
  img_area = morphology.binary_erosion(label, morphology.diamond(D_MARGIN))
  img_area = img_area if len(np.where(img_area == 1)[0]) > 0 else np.copy(label)

  # Set of ground truth pixels.
  O = np.where(img_area == 1)
  # Randomly sample the number of positive clicks and negative clicks to use.
  num_clicks_pos = 0 if len(O) == 0 else random.sample(list(range(1, Npos + 1)), 1)
  # num_clicks_pos = random.sample(range(1, Npos + 1), 1)
  pts = get_sampled_locations(O, img_area, num_clicks_pos)
  u1 = get_distance_transform(pts, img_area, img=img, dt_method=dt_method)

  return u1, pts


def get_neg_dst_transform(label_unmodified, img, instance, strategy, ignore_classes, old_label=None, dt_method="edt"):
  """
  :param img: input image: this would be used to calculate geodesic distance.
  :param ignore_classes: 
  :param dt_method: 'edt' for euclidean distance and 'geodesic' for geodesic distance.
  :param old_label: old label, if available
  :param label_unmodified: unmodified label which contains all the instances
  :param instance: The instance number to segment
  :param strategy: value in [1,2,3]
          1 - Generate random clicks from the background, which is D pixels away from the object.
          2 - Generate random clicks on each negative object.
          3 - Generate random clicks around the object boundary.
  :return: Negative distance transform map
  """
  label = np.where(label_unmodified == instance, 1, 0)
  g_c = get_image_area_to_sample(label)

  pts = []

  if strategy in [1,3]:
    if strategy == 1:
      num_neg_clicks = random.sample(list(range(0, Nneg1 + 1)), 1)
      pts = get_sampled_locations(np.where(g_c == 1), g_c, num_neg_clicks)
    else:
      # First negative click is randomly sampled in g_c
      pts = get_sampled_locations(np.where(g_c == 1), g_c, [1])
      g_c_copy = np.copy(g_c)
      g_c_copy[list(zip(*(val for val in pts)))] = 0
      dt = distance_transform_edt(g_c_copy)
      # Sample successive points using p_next = arg max f(p_ij | s0 U g), where p_ij in g_c, s0 is the set of all
      # sampled points, and 'g' is the complementary set of g_c
      for n_clicks in range(2, Nneg3 + 1):
        if np.max(dt) > 0:
          row, col = np.where(dt == np.max(dt))
          row, col = zip(row, col)[0]
          pts.append((row, col))
          x_min = max(0, row - D)
          x_max = min(row + D, dt.shape[0])
          y_min = max(0, col - D)
          y_max = min(col + D, dt.shape[1])
          dt[x_min:x_max, y_min:y_max] = 0

  elif strategy == 2:
    # Get all negative object instances.
    instances = np.setdiff1d(np.unique(label_unmodified), np.append(instance, ignore_classes))

    num_neg_clicks = random.sample(list(range(0, Nneg2 + 1)), 1)
    for i in instances:
      g_c = np.where(label_unmodified == i)
      label = np.where(label_unmodified == i, 1, 0)
      pts_local = get_sampled_locations(g_c, np.copy(label), num_neg_clicks)
      pts = pts + pts_local

  u0 = get_distance_transform(pts, label, img=img, dt_method=dt_method)

  return u0, pts


def get_distance_transform(pts, label, img=None, dt_method="edt"):
  dt = np.ones_like(label)
  if len(pts) > 0:
    if dt_method == "geodesic" and img is not None:
      # dt = np.where(dt != 0, 1e5, 0)
      dt = geo_dist(img, list(zip(*(val for val in pts))))
    else:
      dt[list(zip(*(val for val in pts)))] = 0
      dt = distance_transform_edt(dt)

    return dt
  else:
    # This is important since we divide it by 255 while normalizing the inputs.
    return dt * 255


def get_sampled_locations(sample_locations, img_area, num_clicks):
  d_step = int(D / 2)
  img = np.copy(img_area)
  pts = []
  for click in range(num_clicks[0]):
    pixel_samples = list(zip(sample_locations[0], sample_locations[1]))
    if len(pixel_samples) > 1:
      [x, y] = random.sample(pixel_samples, 1)[0]
      pts.append([x, y])

      x_min = max(0, x - d_step)
      x_max = min(x + d_step, img.shape[0])
      y_min = max(0, y - d_step)
      y_max = min(y + d_step, img.shape[1])
      img[x_min:x_max, y_min:y_max] = 0

      sample_locations = np.where(img == 1)

  return pts


def get_image_area_to_sample(img):
  """
  calculate set g_c, which has two properties
  1) They represent background pixels 
  2) They are within a certain distance to the object
  :param img: Image that represents the object instance
  """

  #TODO: In the paper 'Deep Interactive Object Selection', they calculate g_c first based on the original object instead
  # of the dilated one.

  # Dilate the object by d_margin pixels to extend the object boundary
  img_area = np.copy(img)
  img_area = morphology.binary_dilation(img_area, morphology.diamond(D_MARGIN)).astype(np.uint8)

  g_c = np.logical_not(img_area).astype(int)
  g_c[np.where(distance_transform_edt(g_c) > D)] = 0

  return g_c


def load_clustering_labels(clustering_path):
  from ReID_net.Log import log
  import pickle
  with open(clustering_path, "rb") as f:
    x = pickle.load(f)
    labels = x["class_labels"]
  print("class labels from", clustering_path, ":", labels, file=log.v1)
  assert labels[0] == "outliers", labels
  clustering_labels = list(labels[1:])
  n_classes = len(clustering_labels)
  label_map = {}
  for idx, label in enumerate(clustering_labels):
    label_map[idx] = {"name": label}
  return clustering_labels, label_map, n_classes
