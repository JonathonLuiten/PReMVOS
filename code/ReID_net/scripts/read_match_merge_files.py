
import os
import time
import pickle
import numpy
import tensorflow as tf

model = "COCO_Similarity_Triplet_Edit"

### LOADING ORIGINAL
print("Loading original pickled output")
output_folder = os.getcwd() + "/" + "forwarded/" + model + "/"
start = time.time()
input_folder_name = output_folder + "clustering/"
num_outputs_already_saved = len(os.listdir(input_folder_name))
input_file_name = input_folder_name + "data" + str(num_outputs_already_saved - 1).zfill(4) + ".pkl"
with open(input_file_name, 'rb') as inputfile:
  results = pickle.load(inputfile)
ys = results["ys"]
orig_tags = results["tags"]
crop_ids = results["crop_ids"]
print("Loaded original, elapsed: ", time.time() - start)



### LOADING ORIGINAL
print("Loading new pickled output")
output_folder = os.getcwd() + "/" + "forwarded/" + model + "/"
start = time.time()
input_folder_name = output_folder + "temp/"
num_outputs_already_saved = len(os.listdir(input_folder_name))
input_file_name = input_folder_name + "temp" + str(num_outputs_already_saved - 1).zfill(4) + ".pkl"
with open(input_file_name, 'rb') as inputfile:
  results = pickle.load(inputfile)
# ys = results["ys"]
new_tags = results
# crop_ids = results["crop_ids"]
print("Loaded new, elapsed: ", time.time() - start)


print(('___'.join(orig_tags[0].split('___')[:-1])))
print(('___'.join(new_tags[0].split('___')[:-1])))
print((orig_tags[0]))
print((new_tags[0]))
print((len(orig_tags)))
print((len(new_tags)))
assert(len(orig_tags)==len(new_tags))

# class_labels_all = [tag.split("___")[-1] for tag in orig_tags]
track_ys = []
track_ims = []
track_classes = []



## Extracting only centroids of tracks
print("extracting")
start = time.time()
# final_tags = numpy.copy(orig_tags)
# final_tags = numpy.empty([len(orig_tags)],dtype="S1000")
final_tags = numpy.empty([len(orig_tags)],dtype=object)
# final_tags = ("test_string",)*len(orig_tags)
orig_tracks, orig_track_idx = numpy.unique(["---".join(tag.split("___")[:-2]) for tag in orig_tags], return_inverse=True)
orig_track_ids = numpy.unique(orig_track_idx)
new_tracks, new_track_idx = numpy.unique(["---".join(tag.split("___")[:-2]) for tag in new_tags], return_inverse=True)
new_track_ids = numpy.unique(new_track_idx)
new_class_labels = [tag.split("___")[-1] for tag in new_tags]
max_num = len(orig_tracks)
for idx, (orig_track, orig_track_id) in enumerate(zip(orig_tracks,orig_track_ids)):
  class_label = next(n for n,c in zip(new_class_labels,new_tracks) if c==orig_track)
  # print class_label
  iidx = numpy.arange(len(orig_track_idx))
  curr_ids = iidx[orig_track_idx == orig_track_id]
  for id in curr_ids:
    final_tags[id] = orig_tags[id]+class_label
  print(final_tags[curr_ids[0]])
  # print(orig_tags[curr_ids[0]])
  print((orig_tags[curr_ids[0]] + class_label))
  if idx%10 == 0:
    print((idx, "/", max_num, time.time()-start))
    start = time.time()

print("extracted elapsed =", time.time() - start)

# new_test_tags = ['___'.join(i.split('___')[:-1]) for i in new_tags]
# o_test_tags = ['___'.join(i.split('___')[:-1]) for i in orig_tags]
#
# final_tags = []
# ids = numpy.arange(len(orig_tags))
# start = time.time()
# for idx,(otag,current_o_tag) in enumerate(zip(orig_tags,o_test_tags)):
#   matching_tag = next(n for n,c in zip(new_tags,new_test_tags) if c==current_o_tag)
#   final_tags.append(matching_tag)
#   if idx%10 == 0:
#     print(idx, time.time()-start)
#     start = time.time()

# final_tags = final_tags.tolist()

print((orig_tags[0]))
print((final_tags[0]))
print((orig_tags[21]))
print((final_tags[21]))
print((orig_tags[42]))
print((final_tags[42]))
print((orig_tags[63]))
print((final_tags[63]))

### Saving
print("Saving pickled output")
start = time.time()
results = {"ys": ys, "tags": final_tags, "crop_ids": crop_ids}
output_folder_name = output_folder + "clustering/"
tf.gfile.MakeDirs(output_folder_name)
num_outputs_already_saved = len(os.listdir(output_folder_name))
output_file_name = output_folder_name + "data" + str(num_outputs_already_saved).zfill(4) + ".pkl"
with open(output_file_name, 'wb') as outputfile:
  pickle.dump(results, outputfile)
print("Saved, elapsed: ", time.time() - start)