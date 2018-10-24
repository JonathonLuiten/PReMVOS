import os
import cv2
import time
import glob
import numpy as np
from PIL import Image

def bbox2(img):
  rows = np.any(img, axis=1)
  cols = np.any(img, axis=0)
  rmin, rmax = np.where(rows)[0][[0, -1]]
  cmin, cmax = np.where(cols)[0][[0, -1]]

  return rmin, rmax, cmin, cmax


anns = '/home/luiten/vision/youtubevos/ytvos_data/train/CleanedAnnotations/'
images = '/home/luiten/vision/youtubevos/ytvos_data/train/JPEGImages/'
output_folder = '/home/luiten/vision/youtubevos/ytvos_data/train/ReID_crops/'

if not os.path.exists(output_folder):
  os.makedirs(output_folder)

current_global_object_id_count = 0
min_object_size = 100  # pixels (10*10)
crf = 1.2  # context region factor
# num_to_gen = 5000

folders = sorted(glob.glob(anns + '*/'))
for folder in folders:
  t = time.time()
  seq = folder.split('/')[-2]

  files = sorted(glob.glob(folder + '*.png'))
  for gg_id,file in enumerate(files):

    gt_img = np.array(Image.open(file))
    img = np.array(Image.open(file.replace(anns,images).replace('.png', '.jpg')))
    curr_ids = np.unique(gt_img)
    curr_ids = curr_ids[curr_ids != 0]

    if gg_id == 0:
      ids = np.unique(gt_img)
      ids = ids[ids != 0]
      # global_ids = ids + current_global_object_id_count - 1
      # current_global_object_id_count += len(ids)
      id_counts = np.zeros_like(ids)


    # if np.all(id_counts == num_to_gen):
    #   break

    for curr_id in curr_ids:
      if curr_id not in ids:
        ids = np.append(ids,curr_id)
        id_counts = np.append(id_counts, 0)
        print("YOYOYOYOYOYOYO")

      if current_global_object_id_count <3360:
        break

      # if id_counts[ids == curr_id] == num_to_gen:
      #   continue
      if np.count_nonzero(gt_img == curr_id) < min_object_size:
        continue
      x1, x2, y1, y2 = bbox2(gt_img == curr_id)

      # im_size = img.shape[:2]
      # nx1 = x1 - 0.5*(x2-x1)*(crf-1.0)
      # ny1 = y1 - 0.5*(y2-y1)*(crf-1.0)
      # nx2 = nx1 + (x2-x1)*(crf-1.0)
      # ny2 = ny1 + (y2-y1)*(crf-1.0)
      # nx1 = int(max(nx1, 0))
      # nx2 = int(min(nx2, im_size[0]))
      # ny1 = int(max(ny1, 0))
      # ny2 = int(min(ny2, im_size[1]))

      # cr = (crf-1) * min(x2 - x1, y2 - y1)

      cr1 = 0.5*(crf - 1) * (x2 - x1)
      cr2 = 0.5*(crf - 1) * (y2 - y1)
      im_size = img.shape[:2]
      nx1 = int(max(x1 - cr1, 0))
      nx2 = int(min(x2 + cr1, im_size[0]))
      ny1 = int(max(y1 - cr2, 0))
      ny2 = int(min(y2 + cr2, im_size[1]))

      cropped_im = img[nx1:nx2, ny1:ny2]
      resized_im = cv2.resize(cropped_im, (128, 128))
      # s# print([x1,x2,y1,y2],[nx1,nx2,ny1,ny2],cr)
      # s# write_dir = output_folder + str(global_ids[ids == curr_id][0]).zfill(4) + '-' + str(id_counts[ids == curr_id][0]).zfill(4) + '.jpg'
      write_dir = output_folder + str(curr_id+current_global_object_id_count - 1).zfill(4) + '-' + str(id_counts[ids == curr_id][0]).zfill(4) + '.jpg'
      print(write_dir)
      cv2.imwrite(write_dir, cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR))
      id_counts[ids == curr_id] += 1

    if gg_id == len(files)-1:
      current_global_object_id_count += len(ids)

  if current_global_object_id_count>3380:
    break

  print('finished', seq, time.time() - t,current_global_object_id_count)