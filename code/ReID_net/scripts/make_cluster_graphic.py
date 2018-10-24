#!/usr/bin/env python

import os
from scipy.misc import imsave, imread
import matplotlib.pyplot as plt
import matplotlib
file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/~~~POST_REORDERING/kitti_raw-final-big-128_14_14-final-reordered/128_014_014/'
# file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/~~~POST_REORDERING/schiphol_final_big-128_14_14-final-reordered/128_014_014/'
# file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/~~~POST_REORDERING/oxford_final_big-128_14_14-final-reordered/128_014_014/'
num = 13
# num = 18
# num = 24
num_per = 10
matplotlib.rcParams.update({'font.size': 9})
f, axarr = plt.subplots(num, num_per,sharex=True, sharey=True)
for i in range(num):
  cf = file + str(i).zfill(3) + '/'
  cfiles = sorted(os.listdir(cf))[:num_per]
  axarr[i,0].set_ylabel("cls" + str(i))
  for j,cfn in enumerate(cfiles):
    im = imread(cf+cfn)
    axarr[i, j].imshow(im)
    axarr[i, j].get_xaxis().set_ticks([])
    axarr[i, j].get_yaxis().set_ticks([])
    axarr[i, j].axis('tight')
plt.subplots_adjust(wspace=0)
plt.show()


# import os
# from scipy.misc import imsave, imread
# import matplotlib.pyplot as plt
# import matplotlib
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/kitti_raw-final-big-128_14_14-final/128_014_014/'
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/schiphol_final_big-128_14_14/128_014_014/'
# file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/oxford_final_big-128_14_14/128_014_014/'
# num = 24
# # num = 13
# # num = 18
# # num_per = 10
# num_per = 5
# matplotlib.rcParams.update({'font.size': 15})
# f, axarr = plt.subplots(num/2, num_per*2+1,sharex=True, sharey=True)
# for i in range(num):
#   if i < num/2:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i,0].set_ylabel("cls" + str(i))
#     for j,cfn in enumerate(cfiles):
#       im = imread(cf+cfn)
#       axarr[i, j].imshow(im)
#       axarr[i, j].get_xaxis().set_ticks([])
#       axarr[i, j].get_yaxis().set_ticks([])
#       axarr[i, j].axis('tight')
#   else:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i-int(float(num)/2)-1, num_per+1].set_ylabel("cls" + str(i-1))
#     for j, cfn in enumerate(cfiles):
#       im = imread(cf + cfn)
#       axarr[i-int(float(num)/2), j+num_per+1].imshow(im)
#       axarr[i-int(float(num)/2), j+num_per+1].get_xaxis().set_ticks([])
#       axarr[i-int(float(num)/2), j+num_per+1].get_yaxis().set_ticks([])
#       axarr[i-int(float(num)/2), j+num_per+1].axis('tight')
# axarr[11,6].set_ylabel("cls" + str(23))
# plt.subplots_adjust(wspace=0)
# plt.show()


# import os
# from scipy.misc import imsave, imread
# import matplotlib.pyplot as plt
# import matplotlib
# file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/kitti_raw-final-big-128_14_14-final/128_014_014/'
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/schiphol_final_big-128_14_14/128_014_014/'
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/oxford_final_big-128_14_14/128_014_014/'
# # num = 24
# num = 13
# # num = 18
# num_per = 10
# # num_per = 5
# matplotlib.rcParams.update({'font.size': 15})
# f, axarr = plt.subplots(num/2+1, num_per*2+1,sharex=True, sharey=True)
# for i in range(num):
#   if i < num/2+1:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i,0].set_ylabel("cls" + str(i))
#     for j,cfn in enumerate(cfiles):
#       im = imread(cf+cfn)
#       axarr[i, j].imshow(im)
#       axarr[i, j].get_xaxis().set_ticks([])
#       axarr[i, j].get_yaxis().set_ticks([])
#       axarr[i, j].axis('tight')
#   else:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i-num/2-1, num_per+1].set_ylabel("cls" + str(i))
#     for j, cfn in enumerate(cfiles):
#       im = imread(cf + cfn)
#       axarr[i-num/2-1, j+num_per+1].imshow(im)
#       axarr[i-num/2-1, j+num_per+1].get_xaxis().set_ticks([])
#       axarr[i-num/2-1, j+num_per+1].get_yaxis().set_ticks([])
#       axarr[i-num/2-1, j+num_per+1].axis('tight')
# plt.subplots_adjust(wspace=0)
# plt.show()


# import os
# from scipy.misc import imsave, imread
# import matplotlib.pyplot as plt
# import matplotlib
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/kitti_raw-final-big-128_14_14-final/128_014_014/'
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/schiphol_final_big-128_14_14/128_014_014/'
# file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/oxford_final_big-128_14_14/128_014_014/'
# num = 24
# # num = 13
# # num = 18
# # num_per = 10
# num_per = 5
# matplotlib.rcParams.update({'font.size': 15})
# f, axarr = plt.subplots(num/4, num_per*4+3,sharex=True, sharey=True)
# for i in range(num):
#   if i < num/4:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i,0].set_ylabel("cls" + str(i))
#     for j,cfn in enumerate(cfiles):
#       im = imread(cf+cfn)
#       axarr[i, j].imshow(im)
#       axarr[i, j].get_xaxis().set_ticks([])
#       axarr[i, j].get_yaxis().set_ticks([])
#       axarr[i, j].axis('tight')
#   elif i < num/2:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i-num/2, num_per+1].set_ylabel("cls" + str(i))
#     for j, cfn in enumerate(cfiles):
#       im = imread(cf + cfn)
#       axarr[i-num/4, j+num_per+1].imshow(im)
#       axarr[i-num/4, j+num_per+1].get_xaxis().set_ticks([])
#       axarr[i-num/4, j+num_per+1].get_yaxis().set_ticks([])
#       axarr[i-num/4, j+num_per+1].axis('tight')
#   elif i < num*3/4:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i-num/2, 2*num_per+2].set_ylabel("cls" + str(i))
#     for j, cfn in enumerate(cfiles):
#       im = imread(cf + cfn)
#       axarr[i-num/2, j+2*num_per+2].imshow(im)
#       axarr[i-num/2, j+2*num_per+2].get_xaxis().set_ticks([])
#       axarr[i-num/2, j+2*num_per+2].get_yaxis().set_ticks([])
#       axarr[i-num/2, j+2*num_per+2].axis('tight')
#   else:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i-num*3/4, 3*num_per+3].set_ylabel("cls" + str(i))
#     for j, cfn in enumerate(cfiles):
#       im = imread(cf + cfn)
#       axarr[i-num*3/4, j+3*num_per+3].imshow(im)
#       axarr[i-num*3/4, j+3*num_per+3].get_xaxis().set_ticks([])
#       axarr[i-num*3/4, j+3*num_per+3].get_yaxis().set_ticks([])
#       axarr[i-num*3/4, j+3*num_per+3].axis('tight')
# plt.subplots_adjust(wspace=0)
# plt.show()

# import os
# from scipy.misc import imsave, imread
# import matplotlib.pyplot as plt
# import matplotlib
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/kitti_raw-final-big-128_14_14-final/128_014_014/'
# # file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/schiphol_final_big-128_14_14/128_014_014/'
# file = '/home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/tests/oxford_final_big-128_14_14/128_014_014/'
# num = 24
# # num = 13
# # num = 18
# num_per = 10
# # num_per = 5
# matplotlib.rcParams.update({'font.size': 15})
# f, axarr = plt.subplots(num/3, num_per*3+2,sharex=True, sharey=True)
# for i in range(num):
#   if i < num/3:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i,0].set_ylabel("cls" + str(i))
#     for j,cfn in enumerate(cfiles):
#       im = imread(cf+cfn)
#       axarr[i, j].imshow(im)
#       axarr[i, j].get_xaxis().set_ticks([])
#       axarr[i, j].get_yaxis().set_ticks([])
#       axarr[i, j].axis('tight')
#   elif i < num*2/3:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i-num*1/3, num_per+1].set_ylabel("cls" + str(i))
#     for j, cfn in enumerate(cfiles):
#       im = imread(cf + cfn)
#       axarr[i-num/3, j+num_per+1].imshow(im)
#       axarr[i-num/3, j+num_per+1].get_xaxis().set_ticks([])
#       axarr[i-num/3, j+num_per+1].get_yaxis().set_ticks([])
#       axarr[i-num/3, j+num_per+1].axis('tight')
#   else:
#     cf = file + str(i).zfill(3) + '/'
#     cfiles = sorted(os.listdir(cf))[:num_per]
#     axarr[i-num*2/3, 2*num_per+2].set_ylabel("cls" + str(i))
#     for j, cfn in enumerate(cfiles):
#       im = imread(cf + cfn)
#       axarr[i-num*2/3, j+2*num_per+2].imshow(im)
#       axarr[i-num*2/3, j+2*num_per+2].get_xaxis().set_ticks([])
#       axarr[i-num*2/3, j+2*num_per+2].get_yaxis().set_ticks([])
#       axarr[i-num*2/3, j+2*num_per+2].axis('tight')
# plt.subplots_adjust(wspace=0)
# plt.show()
