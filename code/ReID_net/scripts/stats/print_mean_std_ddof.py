#!/usr/bin/env python
import sys
import numpy

#vals = []
#for l in sys.stdin:
#  vals.append(float(l))
vals = numpy.loadtxt(sys.stdin)

print(vals.size, "mean", numpy.mean(vals), "std", numpy.std(vals, ddof=1))
