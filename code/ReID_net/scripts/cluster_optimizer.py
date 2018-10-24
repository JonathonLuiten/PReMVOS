import os
import time
import pickle
import operator
import numpy as np

model = "COCO_Similarity_Triplet_Edit"
output_folder = os.getcwd() + "/" + "forwarded/" + model + "/"
# num_outputs_already_saved = len(os.listdir(output_folder + "tests/"))
# num_outputs_already_saved = 2
num_outputs_already_saved = 10
test_output_file = output_folder + "tests/" + str(num_outputs_already_saved-1).zfill(3) + "/"

# input_file_name = output_folder + "goods.pkl"
# with open(input_file_name, 'rb') as inputfile:
#   to_use = cPickle.load(inputfile)

start = time.time()
count = 0
number_allowable_outliers = 10
goods = []
# for n_components,min_samples,min_cluster_size in to_use:
for n_components in [16]:
  for min_samples in [16]:
    for min_cluster_size in [10]:
# for n_components in [4,8,12,16,24,32,48,64,96,128]:
#   for min_samples in range(2, 51):
#     for min_cluster_size in range(2, 51):

# for n_components in [8, 12, 16, 24, 32, 48, 64, 96, 128]:
#   for min_samples in range(6,31,2):
#     for min_cluster_size in range(6, 31,2):

      final_output_fol = test_output_file + str(n_components).zfill(3) + "_" + str(min_samples).zfill(3) + "_" + str(min_cluster_size).zfill(3) + "/"
      text_file_name = final_output_fol + "summary.txt"
      F = open(text_file_name, 'r')
      F.readline()
      F.readline()
      F.readline()
      F.readline()
      F.readline()
      F.readline()
      num_classes = int(F.readline().split(': ')[-1])
      num_outliers = int(F.readline().split(': ')[-1])
      F.readline()
      F.readline()
      F.readline()
      class_counts = []
      names = []
      cluster_outliers = []
      bad = False
      for i in range(num_classes+1):
        strs = (F.readline().split(', ')[2:])
        curr_list = [(cstr.split(': ')[0],int(cstr.split(': ')[1])) for cstr in strs]
        curr_dict = dict(curr_list)
        u_count = curr_dict.get('unknown_type',0) + curr_dict.get('unknown',0)
        curr_dict['unknown'] = u_count
        curr_dict.pop("unknown_type", None)
        curr_dict_without_unknown = curr_dict.copy()
        curr_dict_without_unknown.pop('unknown',None)
        if i == 0:
          name = 'outliers'
          group_outliers = 0
        elif len(curr_dict) == 1:
          name = list(curr_dict.keys())[0]
          group_outliers = 0
        elif len(curr_dict_without_unknown) == 1:
          name = list(curr_dict_without_unknown.keys())[0]
          group_outliers = 0
        else:
          labels = list(curr_dict_without_unknown.keys())
          label_counts = list(curr_dict_without_unknown.values())
          cidx = np.argmax(label_counts)
          main, main_count = (labels[cidx],label_counts[cidx])
          curr_dict_without_unknown_main = curr_dict_without_unknown.copy()
          curr_dict_without_unknown_main.pop(main, None)
          group_outliers = sum(list(curr_dict_without_unknown_main.values()))
          if main_count>group_outliers:
            name = main
          else:
            name = 'unknown'

        class_counts.append(curr_dict)
        names.append(name)
        cluster_outliers.append(group_outliers)

      required_outliers = max(cluster_outliers)
      # names_without_unknown = [a for a in names if a != 'unknown']
      # all_names,name_inverses,name_count = np.unique(names_without_unknown, return_inverse=True,return_counts=True)
      all_names, name_inverses, name_count = np.unique(names, return_inverse=True, return_counts=True)
      all_ids = np.arange(len(name_inverses))
      extra_counts = []
      for a_idx, (a_name, a_count) in enumerate(zip(all_names,name_count)):
        if a_count>1 and a_name not in 'unknown':
          ids = all_ids[name_inverses == a_idx]
          counts_list = [class_counts[i][a_name] for i in ids]
          counts_list.sort(reverse=True)
          extra_counts_list = counts_list[1:]
          extra_counts.append(sum(extra_counts_list))
      if extra_counts:
        tot_extra = max(extra_counts)
      else:
        tot_extra = 0
      if required_outliers<tot_extra:
        required_outliers = tot_extra
      if required_outliers>=min_cluster_size:
        required_outliers+=1000
      for ii,(cc,nn) in enumerate(zip(class_counts,names)):
        if ii>0 and (cc[nn]<=required_outliers or cc[nn]<0.5*sum(list(cc.values()))):
          names[ii] = 'unknown'

      all_names, name_inverses, name_count = np.unique(names, return_inverse=True, return_counts=True)
      for a_idx, (a_name, a_count) in enumerate(zip(all_names, name_count)):
        if a_count > 1:
          ids = all_ids[name_inverses == a_idx]
          for ii,id in enumerate(ids):
            names[id] = names[id]+'_'+str(ii)

      if required_outliers <= number_allowable_outliers:
        count += 1
        print(n_components, min_samples, min_cluster_size, num_classes, num_outliers, required_outliers, names)
        goods.append((n_components, min_samples, min_cluster_size))

print(time.time()-start, count)
#
# output_file_name = output_folder + "goods.pkl"
# with open(output_file_name, 'wb') as outputfile:
#   cPickle.dump(goods, outputfile)