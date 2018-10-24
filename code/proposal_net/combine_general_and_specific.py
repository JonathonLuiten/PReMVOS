#!/usr/bin/env python

import time
import glob
import json
import os

root_dir = "./output/intermediate/"
one_dir = 'general_proposals/'
two_dir = 'specific_proposals/'
out_dir = 'combined_proposals/'

t = time.time()
files = sorted(glob.glob(root_dir  + one_dir + '*/*.json'))
translated_files = [file.replace(one_dir,two_dir) for file in files]
second_files = sorted(glob.glob(root_dir  + two_dir + '*/*.json'))
to_add_files = [file for file in second_files if file not in translated_files]
if to_add_files:
  print(to_add_files)
files += to_add_files
print(time.time()-t)
for file in files:
  two_file = file.replace(one_dir,two_dir)
  out_file = file.replace(one_dir,out_dir)
  try:
    with open(file, 'r') as f:
      proposals1 = json.load(f)
  except:
    proposals1 = []
  try:
    with open(two_file, 'r') as f:
      proposals2 = json.load(f)
  except:
    proposals2 = []
  fin_proposals = proposals1+proposals2
  outfolder = '/'.join(out_file.split('/')[:-1])
  if not os.path.exists(outfolder):
    os.makedirs(outfolder)
  with open(out_file, "w") as f:
    json.dump(fin_proposals,f)

print(time.time()-t)