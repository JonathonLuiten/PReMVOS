#!/usr/bin/env python

import os
import sys
import json

if __name__ == "__main__":
  assert len(sys.argv) in (2, 3)
  folder = "forwarded/" + sys.argv[1] + "/"
  if len(sys.argv) == 3:
    json_txt = open(folder + sys.argv[2] + ".json").read()
  else:
    json_txt = open(folder + "forwarded.json").read()
  dets = json.loads(json_txt)

  try:
    os.mkdir(folder + "/data/")
  except:
    pass
  opened = None
  for det in dets:
    filename = folder + "/data/" + det["image_id"].split("/")[-1].replace(".png", ".txt")
    if opened is None:
      opened = (filename, open(filename, "w"))
    else:
      if filename != opened[0]:
        opened[1].close()
        opened = (filename, open(filename, "w"))
    class_ = det["category_id"]
    if class_ == 1:
      class_str = "Pedestrian"
    elif class_ == 2:
      class_str = "Cyclist"
    elif class_ == 3:
      class_str = "Car"
    else:
      print("warning, ignoring detection with unknown class id", class_)
      continue
    bbox = det["bbox"]
    score = det["score"]
    print(class_str, 0, 0, -10, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], 0, 0, 0, 0, 0, \
        0, 0, score, file=opened[1])
  if opened is not None:
    opened[1].close()
