#!/usr/bin/env python

import sys
import numpy

assert len(sys.argv) == 2
fn = sys.argv[1]

ious = [None] * 20

with open(fn) as f:
  for l in f:
    if "sequence" in l:
      sp = l.split()
      idx = int(sp[1]) - 1
      iou = float(sp[-1])
      assert 0 <= iou <= 1
      ious[idx] = iou

for i in range(10):
  assert ious[i] is not None, i

if ious[10] is None:
  for i in range(10, 20):
    assert ious[i] is None, i
  ious = ious[:10]
else:
  for i in range(10, 20):
    assert ious[i] is not None, i

print(len(ious), numpy.mean(ious))
