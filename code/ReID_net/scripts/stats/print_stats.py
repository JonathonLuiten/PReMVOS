#!/usr/bin/env python
import sys
import numpy

vals = []
for l in sys.stdin:
  vals.append(float(l))

print(len(vals), "mean", numpy.mean(vals), "std", numpy.std(vals), "min", numpy.min(vals), "max", numpy.max(vals))
