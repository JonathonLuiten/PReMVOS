#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys


def doit(fn, col1, col2, tag):
  train = []
  val = []
  with open(fn) as f:
    for l in f:
      if "finished" in l:
        sp = l.split()
        #clip to 5
        # tr = min(float(sp[col1]), 5.0)
        # va = min(float(sp[col2]), 5.0)
        tr = float(sp[col1])
        va = float(sp[col2])
        # if tr>1: tr = 1/tr
        # if va>1: va = 1/va

        train.append(tr)
        val.append(va)
  plt.plot(train, label="train")
  plt.hold(True)
  plt.plot(val, label="val")
  plt.legend()
  plt.title(fn + " " + tag)
  #plt.show()

assert len(sys.argv) == 2
#error plot
doit(sys.argv[1], 6, 8, "rank1") #For plotting for extended
# doit(sys.argv[1], 21, 17, "rank1") #For plotting for triplet
#score plot
# plt.figure()
# doit(sys.argv[1], 22, 18, "rank5")
plt.show()
