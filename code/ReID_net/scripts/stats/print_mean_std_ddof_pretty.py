#!/usr/bin/env python
import sys
import numpy

#vals = []
#for l in sys.stdin:
#  vals.append(float(l))
vals = numpy.loadtxt(sys.stdin)

assert vals.size == 3
print("%.2f +- %.2f" % (numpy.mean(vals) * 100, numpy.std(vals, ddof=1) * 100))
