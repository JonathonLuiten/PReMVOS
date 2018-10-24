from skimage.transform import SimilarityTransform, warp
from .Video import shift
import numpy
from math import sqrt
from scipy.ndimage import grey_dilation


def shift_mask_speed_absolute(mask, max_speed=30):
  speed = (numpy.random.rand() * 2 - 1.0) * max_speed
  offset = numpy.random.rand(2)
  offset /= numpy.linalg.norm(offset, 2)
  offset *= speed
  return shift(mask, offset, "constant", 0)


#factor w.r.t. object size
def shift_mask_speed_factor(mask, factor=0.1):
  #this method of determining the object size seems to be bad!
  #size = sqrt(mask.sum())

  nzy, nzx, _ = mask.nonzero()
  if nzy.size == 0:
    return mask
  size = sqrt((nzy.max() - nzy.min()) * (nzx.max() - nzx.min()))
  max_speed = int(round(factor * size))
  #print max_speed
  return shift_mask_speed_absolute(mask, max_speed)


def scale_mask(mask, factor=1.05):
  nzy, nzx, _ = mask.nonzero()
  if nzy.size == 0:
    return mask
  #center_y, center_x = nzy.mean(), nzx.mean()
  #print center_y, center_x
  center_y, center_x = (nzy.max() + nzy.min()) / 2, (nzx.max() + nzx.min()) / 2
  #print center_y, center_x

  shift_ = SimilarityTransform(translation=[-center_x, -center_y])
  shift_inv = SimilarityTransform(translation=[center_x, center_y])

  A = SimilarityTransform(scale=(factor, factor))
  mask_out = warp(mask, (shift_ + (A + shift_inv)).inverse)
  mask_out = (mask_out > 0.5).astype("float32")
  #import matplotlib.pyplot as plt
  #im = numpy.concatenate([mask, mask, mask_out],axis=2)
  #plt.imshow(im)
  #plt.show()
  return mask_out


def damage_mask(mask, scale_factor, shift_absolute, shift_factor):
  assert not (shift_absolute != 0.0 and shift_factor != 0.0)
  mask = mask.astype("float32")

  if shift_absolute != 0.0:
    mask = shift_mask_speed_absolute(mask, max_speed=shift_absolute)
  elif shift_factor != 0.0:
    mask = shift_mask_speed_factor(mask, factor=shift_factor)

  if scale_factor != 0.0:
    scale = numpy.random.uniform(1 - scale_factor, 1 + scale_factor)
    mask = scale_mask(mask, scale)

  # dilation_size = 5
  # mask = grey_dilation(mask, size=(dilation_size, dilation_size, 1))
  return mask
