import os
import random
from collections import defaultdict
from gc import get_objects
from scipy.misc import imread
from scipy.ndimage.morphology import distance_transform_edt

import numpy as np
from skimage import measure as skimage_measure

import ReID_net.Constants as Constants
import ReID_net.Measures as Measures
import ReID_net.datasets.Util.Util as Util
from ReID_net.Log import log
from ReID_net.datasets.Util.Util import get_masked_image

before = defaultdict(int)
after = defaultdict(int)


class InteractiveEval(object):
  def __init__(self, engine,  mask_generation_fn):
    self.engine = engine
    self.data = engine.valid_data
    self.mask_generation_fn = mask_generation_fn
    self.max_clicks = engine.config.int("max_clicks", 8)
    self.save_plot = engine.config.int("save_plot", False)
    self.neg_row = []
    self.neg_col = []
    self.pos_row = []
    self.col_pos = []
    #Minimus distance between the clicks
    self.d_step = 10

  def create_distance_transform(self, label):
    u0 = Util.get_distance_transform(list(zip(self.neg_row, self.neg_col)), label)
    u1 = Util.get_distance_transform(list(zip(self.pos_row, self.col_pos)), label)

    # Add extra channels that represent the clicks
    u0 = u0[:, :, np.newaxis]
    click_channel = np.zeros_like(u0)
    click_channel[self.neg_row, self.neg_col] = 1
    u0 = np.concatenate([u0, click_channel.astype(np.float32)], axis=2)
    u1 = u1[:, :, np.newaxis]
    click_channel = np.zeros_like(u1)
    click_channel[self.pos_row, self.col_pos] = 1
    u1 = np.concatenate([u1, click_channel.astype(np.float32)], axis=2)

    return u0, u1

  def eval(self):
    input_list = self.data.read_inputfile_lists()
    input_list = list(zip(input_list[0], input_list[1]))
    measures = {}
    count = 0

    for im, an in input_list:
      im_path = im.split(":")[0]
      file_name = im_path.split("/")[-1]
      file_name_without_ext = file_name.split(".")[0]
      an_path = an.split(":")[0]
      inst = int(an.split(":")[1])

      if os.path.exists(an_path):
        label_unmodified = imread(an_path)
        img_unmodified = imread(im_path)
        self.neg_row = []
        self.neg_col = []
        self.pos_row = []
        self.col_pos = []
        label = np.where(label_unmodified == inst, 1, 0)

        if len(np.where(label_unmodified == inst)[0]) < 2500:
          continue
        count += 1
        img, label = self.create_inputs(img_unmodified, label)
        mask = None
        click_added=True
        clicks = 1
        # Add a positive click when there are no previous masks
        self.add_clicks(mask, label)
        u0, u1 = self.create_distance_transform(label)
        while clicks <= self.max_clicks:
          if self.save_plot:
            file_name = file_name_without_ext + "_instance_" + repr(inst) + "_clicks_" + repr(clicks)
            self.save_image(file_name, img_unmodified, mask)
          #break and continue with the next instance, if a click could not be added
          if not click_added:
            break

          print(repr(count) + "/" + repr(len(input_list)) + "-- Forwarding File:" + \
                           im + " Instance: " + repr(inst) + " Clicks:" + repr(clicks), file=log.v5)

          for i in get_objects():
            before[type(i)] += 1

          mask, new_measures = self.mask_generation_fn(img, tag=im, label=label[:, :, np.newaxis],old_label=None,
                                                       u0=u0, u1=u1)

          # leaked_things = [[x] for x in range(10)]
          # for i in get_objects():
          #   after[type(i)] += 1
          # print [(k, after[k] - before[k]) for k in after if after[k] - before[k]]

          if clicks in measures:
            measures[clicks] += [new_measures]
          else:
            measures[clicks] = [new_measures]

          click_added = self.add_clicks(mask, label)
          u0, u1 = self.create_distance_transform(label)
          clicks += 1

    x_val = []
    y_val = []
    for click in measures:
      avg_measure = Measures.average_measures(measures[click])
      if Constants.IOU in avg_measure:
        x_val.append(click)
        y_val.append(float(avg_measure[Constants.IOU]))
      print("Average measure for " + repr(click) + " clicks: " + repr(avg_measure), file=log.v5)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(x_val, y_val)
    plt.savefig(self.get_file_path("eval_plot"))

  def add_clicks(self, mask, label):
    if mask is None:
      row, col = self.generate_click(label, 1)
      if row is None or col is None:
        print("The label is empty.")
      else:
        self.pos_row.append(row)
        self.col_pos.append(col)
        return True
    else:
      # Find the misclassified pixels.
      I = np.logical_and(mask, label).astype(np.int)
      U = np.logical_or(mask, label).astype(np.int)
      misclassified = np.abs(U - I)
      #       TODO: find false positives and false negatives, and add the clicks accordingly.
      #       Label the misclassified clusters using connected components, and take the largest cluster.
      misclassified = skimage_measure.label(misclassified, background=0)
      clusters = np.setdiff1d(np.unique(misclassified), [0])
      if len(clusters) > 0:
        cluster_lengths = {len(np.where(misclassified == cluster)[0]): cluster for cluster in clusters}
        # Repeat until all possible clusters are sampled
        while len(list(cluster_lengths.keys())) > 0:
          cluster = cluster_lengths[max(cluster_lengths.keys())]
          # index = cluster_lengths.index(max(cluster_lengths))
          row, col = self.generate_click(misclassified, cluster)

          # If a click cannot be added to the current cluster, then move on to the next cluster.
          if row is not None and col is not None:
            # Check if the sampled pixel lies on the object
            if label[row, col] == 1:
              self.pos_row.append(row)
              self.col_pos.append(col)
            else:
              self.neg_row.append(row)
              self.neg_col.append(col)
            return True
          else:
            # Remove the current cluster, so that the next largest cluster can be sampled.
            del cluster_lengths[max(cluster_lengths.keys())]

    return False

  def generate_click(self, mask, inst):
    dt = np.where(mask == inst, 1, 0)
    # Set the current click points to 0, so that a reasonable new sample is obtained.
    dt[self.pos_row, self.col_pos] = 0
    dt[self.neg_row, self.neg_col] = 0
    
    #Set the border pixels of the image to 0, so that the click is centred on the required mask.
    dt[[0,dt.shape[0] - 1], : ] = 0
    dt[:, [0, dt.shape[1] - 1]] = 0

    dt = distance_transform_edt(dt)
    row = None
    col = None

    if np.max(dt) > 0:
      # get points that are farthest from the object boundary.
      #farthest_pts = np.where(dt > np.max(dt) / 2.0) 
      farthest_pts = np.where(dt == np.max(dt))
      # sample from the list since there could be more that one such points.
      row, col = random.sample(list(zip(farthest_pts[0], farthest_pts[1])), 1)[0]

      #Calculate distance from the existing clicks, and ignore if it is within d_step distance.
      dt_pts = np.ones_like(dt)
      dt_pts[self.pos_row, self.col_pos] = 0
      dt_pts[self.neg_row, self.neg_col] = 0
      dt_pts = distance_transform_edt(dt_pts)

      if dt_pts[row, col] < self.d_step:
        row = None
        col = None

    return row, col

  def save_image(self, file_name, img, mask):
    if mask is None:
      mask = np.zeros_like(img[:, :, 0])

    masked_img = get_masked_image(img, mask)
    file_path = self.get_file_path(file_name)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(masked_img)
    plt.scatter(x=self.neg_col, y=self.neg_row, c='r', s=40)
    plt.scatter(x=self.col_pos, y=self.pos_row, c='g', s=40)
    plt.savefig(file_path)
    plt.close()

  def get_file_path(self, file_name):
    main_folder = "forwarded/" + self.engine.model + "/valid/plots/"
    if not os.path.exists(main_folder):
      os.makedirs(main_folder)
    file_path = main_folder + file_name + ".png"
    return file_path

  @staticmethod
  def create_inputs(img, label):
    return img / 255.0, label






