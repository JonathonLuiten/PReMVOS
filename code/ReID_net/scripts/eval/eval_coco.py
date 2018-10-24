#!/usr/bin/env python

import sys
import pycocotools.coco as coco
import pycocotools.cocoeval as cocoeval

if len(sys.argv) < 2 or len(sys.argv) > 3:
  print("usage: {} det_file.json [gt_file.json]".format(sys.argv[0]))
  sys.exit(1)
det_file = sys.argv[1]
if len(sys.argv) == 3:
  gt_file = sys.argv[2]
else:
  gt_file = "/fastwork/voigtlaender/mywork/data/coco/annotations/instances_valid.json"

coco_gt = coco.COCO(gt_file)
coco_det = coco_gt.loadRes(det_file)
e = cocoeval.COCOeval(coco_gt, coco_det, "bbox")
e.evaluate()
e.accumulate()
e.summarize()
