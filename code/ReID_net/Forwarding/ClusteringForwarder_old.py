#from twisted.application.internet import _AbstractClient

import tensorflow as tf
import numpy
import os
from scipy.misc import imsave, imread, imresize
import pickle
import time
from sklearn.externals.joblib import Memory
from sklearn.decomposition import PCA
import sys
sys.setrecursionlimit(10000)
# from hdbscan import HDBSCAN,all_points_membership_vectors, approximate_predict

from .Forwarder import Forwarder
from ReID_net.Log import log

class ClusteringForwarder(Forwarder):
  def __init__(self, engine):
    super(ClusteringForwarder, self).__init__(engine)
    self.clustering_method = self.config.str("clustering_algorithm", "")
    assert self.clustering_method in ("argmax", "kmeans", "dbscan", "hdbscan")
    self.num_clusters = self.config.int("num_clusters", -1)
    if self.clustering_method == "kmeans":
      assert self.num_clusters != -1
    self.use_pre_saved_data = self.config.bool("use_pre_saved_data", False)
    self.pre_saved_to_use = self.config.str("pre_saved_to_use", "")
    self.use_pre_extracted = self.config.bool("use_pre_extracted", False)
    self.output_folder = os.getcwd() + "/" + "forwarded/" + self.model + "/"
    self.do_cluster_all_options = self.config.bool('do_cluster_all_options', True)
    self.do_combine = self.config.bool('do_combine',False)
    self.do_optimize_params = self.config.bool('do_optimize_params',False)
    self.clustering_meta_params = self.config.int_list('clustering_meta_params',None)
    self.exp_name = self.pre_saved_to_use + '-' + '_'.join(map(str, self.clustering_meta_params))
    self.clusterer_to_use = self.config.str('clusterer_to_use',"") + '-' + '_'.join(map(str, self.clustering_meta_params))
    self.s=32

  def get_all_latents_from_network(self):
    # num_outputs_already_saved = len(os.listdir(self.output_folder + "crops/"))
    # self.crop_folder = self.output_folder + "crops/" + str(num_outputs_already_saved).zfill(3) + "/"
    self.crop_folder = self.output_folder + "crops/" + self.pre_saved_to_use + "/"
    # y_softmax = self.network.y_softmax
    y_softmax = self.network.get_output_layer().outputs
    tag = self.network.tags
    img_raw = self.network.inputs_tensors_dict["imgs_raw"]
    original_labels = self.network.inputs_tensors_dict["original_labels"]

    n_total = self.data.num_examples_per_epoch()
    tf.gfile.MakeDirs(self.crop_folder)

    n_processed = 0
    ys = []
    crop_ids = []
    tags = []
    labels = []
    while n_processed < n_total:
      start = time.time()
      y_val, img_raw_val, tag_val,label_val = self.session.run([y_softmax, img_raw, tag,original_labels])
      y_val = y_val[0]
      curr_size = y_val.shape[0]
      for i in range(curr_size):
        ys.append(y_val[i])
        tags.append(tag_val[i])
        labels.append(label_val[i])

        ## Save crops
        idx = n_processed + i
        # crop_path = self.crop_folder + ("%07d" % idx) + ".jpg"
        crop_path = self.crop_folder + ("%07d" % idx) + ".jpg"
        # im = imresize(img_raw_val[i], [self.s, self.s])
        im = img_raw_val[i]
        imsave(crop_path, im)
        crop_ids.append(idx)

      n_processed += curr_size
      print(n_processed, "/", n_total, " elapsed: ", time.time() - start, file=log.v5)

    ### Save vectors and tags
    ys = numpy.array(ys)
    print("Saving pickled output", file=log.v5)
    start = time.time()
    results = {"ys": ys, "tags": tags, "crop_ids": crop_ids, "labels":labels}
    output_folder_name = self.output_folder + "clustering/"
    tf.gfile.MakeDirs(output_folder_name)
    # num_outputs_already_saved = len(os.listdir(output_folder_name))
    # output_file_name = output_folder_name + "data" + str(num_outputs_already_saved).zfill(4) + ".pkl"
    output_file_name = output_folder_name + self.pre_saved_to_use + ".pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)
    return ys, tags, crop_ids,labels

  def get_all_latents_from_faster_rcnn(self):
    output_layer = self.network.get_output_layer()
    roi_placeholder = output_layer.roi_placeholder
    scores = output_layer.classification_outputs
    features = output_layer.classification_features
    data = self.engine.valid_data
    #bboxes = self.network.inputs_tensors_dict["bboxes"]
    #ignore_regions = self.network.inputs_tensors_dict["ignore_regions"]
    #ids = self.network.inputs_tensors_dict["ids"]
    tags = self.network.inputs_tensors_dict["tags"]
    #n_boxes = tf.reduce_sum(tf.cast(ids > 0, tf.int32))
    #extract_boxes = bboxes[0, :n_boxes]

    #feed_dict = {roi_placeholder: extract_boxes}
    n_examples = data.num_examples_per_epoch()
    # n_examples = 500
    n_processed = 0
    out_features = []
    out_tags = []
    out_hyps = []
    out_ts = []
    out_scores = []
    out_bbs = []
    out_labels = []
    out_tracklet_filenames = []
    while n_processed < n_examples:
      start =time.time()
      scores_val, features_val, tags_val = self.engine.session.run([scores, features, tags])
      tag_val = tags_val[0]
      if self.config.str("dataset", "") == "tracklet_detection":
        # print features_val.shape
        anns = data.filename_to_anns[tag_val]
        # print "scores shape", scores_val.shape
        # print len(anns), len(features_val), scores_val.shape[0]
        assert len(anns) == len(features_val) == scores_val.shape[0]
        for ann, feat, score in zip(anns, features_val, scores_val):
          t = ann["time"]
          hyp_idx = ann["hyp_idx"]
          bbox = ann["bbox"]
          label = ann["category_id"]
          tracklet_filename = ann["tracklet_filename"]
          # print feat.shape, tags_val, hyp_idx, t, score.argmax(), bbox
          out_features.append(feat)
          out_tags.append(tag_val)
          out_hyps.append(hyp_idx)
          out_ts.append(t)
          out_scores.append(score)
          out_bbs.append(bbox)
          out_labels.append(label)
          out_tracklet_filenames.append(tracklet_filename)
      else:
        tag_val_new = tag_val.split("/")[-1]
        anns = data.filename_to_anns[tag_val_new]
        for idx, (ann, feat, score) in enumerate(zip(anns, features_val, scores_val)):
          bbox = [int(round(x)) for x in ann["bbox"]]
          label = ann["category_id"]
          # tracklet_filename = ann["tracklet_filename"]
          # print feat.shape, tags_val, score.argmax(), bbox
          out_features.append(feat)
          out_tags.append(tag_val)
          out_hyps.append(idx)
          t = 0
          out_ts.append(t)
          out_scores.append(score)
          out_bbs.append(bbox)
          out_labels.append(label)
          # out_tracklet_filenames.append(tracklet_filename)
      n_processed += 1
      print(n_processed, "/", n_examples, "elapsed:", time.time()-start)

    print(len(out_features), len(out_tags), len(out_hyps), len(out_ts), len(out_scores), len(out_bbs), len(out_labels), len(out_tracklet_filenames))
    ## Saving file of tracks to cluster labels
    print("Saving pickled output", file=log.v5)
    start = time.time()
    results = {"features": out_features, "tags": out_tags, "hyps": out_hyps, "ts": out_ts, "scores":out_scores, "bboxes":out_bbs, "labels":out_labels, "tracklet_filename":out_tracklet_filenames}
    output_folder_name = self.output_folder + "RCNN_forward/"
    tf.gfile.MakeDirs(output_folder_name)
    if self.config.str("dataset", "") == "tracklet_detection":
      output_file_name = output_folder_name + "KITTI" + ".pkl"
    else:
      output_file_name = output_folder_name + "COCO" + ".pkl"
    print(output_file_name)
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)


    quit()
    return

  def get_RCNN_latents_from_file(self):
    ### Load vectors and tags
    print("Loading pickled output", file=log.v5)
    start = time.time()
    input_file_name = self.output_folder + "RCNN_forward/COCO_final.pkl"
    with open(input_file_name, 'rb') as inputfile:
      results = pickle.load(inputfile)
    ys = results["features"]
    tags = results["tags"]
    crop_ids = {"hyps":results["hyps"],"ts":results["ts"],"bboxes":results["bboxes"],"tracklet_filenames":results["tracklet_filename"]}
    labels = results["labels"]
    print(len(labels))
    print("Loaded, elapsed: ", time.time() - start, file=log.v5)
    return ys, tags, crop_ids,labels

  def get_all_latents_from_file(self):
    ### Load vectors and tags
    num_outputs_already_saved = len(os.listdir(self.output_folder + "crops/"))
    if self.pre_saved_to_use in "":
      self.crop_folder = self.output_folder + "crops/" + str(num_outputs_already_saved - 1).zfill(3) + "/"
    else:
      self.crop_folder = self.output_folder + "crops/" + self.pre_saved_to_use + "/"
    print("Loading pickled output", file=log.v5)
    start = time.time()
    input_folder_name = self.output_folder + "clustering/"
    if self.pre_saved_to_use in "":
      num_outputs_already_saved = len(os.listdir(input_folder_name))
      input_file_name = input_folder_name + "data" + str(num_outputs_already_saved - 1).zfill(4) + ".pkl"
    else:
      input_file_name = input_folder_name + self.pre_saved_to_use + ".pkl"
    with open(input_file_name, 'rb') as inputfile:
      results = pickle.load(inputfile)
    ys = results["ys"]
    tags = results["tags"]
    crop_ids = results["crop_ids"]
    labels = results["labels"]
    # labels = None
    print("Loaded, elapsed: ", time.time() - start, file=log.v5)
    return ys, tags, crop_ids,labels

  def extract_centroids(self,ys,tags,labels):
    ## Extracting only centroids of tracks
    print("extracting centroids", file=log.v5)
    start = time.time()
    track_tags_name, track_idx = numpy.unique(["---".join(tag.split("___")[:-2]) for tag in tags], return_inverse=True)
    print(track_tags_name)
    track_tags = numpy.unique(track_idx)
    class_labels_all = [tag.split("___")[-1] for tag in tags]
    track_ys = []
    track_ims = []
    track_classes = []
    max_classes = []
    track_crop_ids = []
    center_ids = []
    track_labels = []
    for track_tag in track_tags:
      iidx = numpy.arange(len(track_idx))
      curr_ids = iidx[track_idx == track_tag]
      curr_avg = numpy.average(ys[curr_ids], axis=0)
      dists = numpy.sqrt(numpy.sum((ys[curr_ids] - curr_avg) ** 2, axis=1))
      idx = curr_ids[numpy.argmin(dists)]
      crop_path = self.crop_folder + ("%07d" % idx) + ".jpg"
      track_crop_ids.append(idx)
      track_ys.append(ys[idx])
      # track_ys.append(curr_avg)
      # track_ims.append(crop_path)
      # im = imread(crop_path)
      # track_ims.append(im)
      curr_class_labels = [class_labels_all[x] for x in curr_ids]

      # New Max
      unique_curr_labels, unique_curr_labels_counts = numpy.unique(curr_class_labels, return_counts=True)
      max_id = numpy.argmax(unique_curr_labels_counts)
      max_curr_label = unique_curr_labels[max_id]


      ## OLD MAX
      # curr_class_labels_without_unknown = [x for x in curr_class_labels if x != 'unknown' and x != 'unknown_type']
      # if len(curr_class_labels_without_unknown)<1:
      #   max_curr_label = 'unknown'
      # else:
      #   unique_curr_labels,unique_curr_labels_counts = numpy.unique(curr_class_labels_without_unknown,return_counts=True)
      #   if len(unique_curr_labels)==1:
      #     max_curr_label = unique_curr_labels[0]
      #   else:
      #     print "WARNING, track has multiple hard labels", [(l,c) for l,c in zip(unique_curr_labels,unique_curr_labels_counts)]
      #     max_id = numpy.argmax(unique_curr_labels_counts)
      #     max_curr_label = unique_curr_labels[max_id]

      track_classes.append(class_labels_all[idx])
      max_classes.append(max_curr_label)
      center_ids.append(idx)
      track_labels.append(labels[idx])
    original_track_ys = numpy.array(track_ys)
    print("centroids extracted elapsed =", time.time() - start, file=log.v5)

    # track_ims = [imresize(im, [self.s, self.s]) for im in track_ims]

    ## Save extracted
    print("Saving pickled output", file=log.v5)
    start = time.time()
    results = {"track_ys": track_ys, "track_tags_name": track_tags_name, "track_crop_ids": track_crop_ids,
               "track_classes": track_classes, "max_classes": max_classes, "center_ids":center_ids, "track_labels":track_labels}
    output_folder_name = self.output_folder + "extracted/"
    tf.gfile.MakeDirs(output_folder_name)
    # num_outputs_already_saved = len(os.listdir(output_folder_name))
    # output_file_name = output_folder_name + "extracted" + str(num_outputs_already_saved).zfill(4) + ".pkl"
    output_file_name = output_folder_name + self.pre_saved_to_use + ".pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)

    return original_track_ys, track_ims, track_classes, track_crop_ids, track_tags_name, max_classes, center_ids, track_labels

  def load_pre_extracted_centroids(self):
    ### Load vectors and tags
    num_outputs_already_saved = len(os.listdir(self.output_folder + "crops/"))
    if self.pre_saved_to_use in "":
      crop_folder = self.output_folder + "crops/" + str(num_outputs_already_saved - 1).zfill(3) + "/"
    else:
      crop_folder = self.output_folder + "crops/" + self.pre_saved_to_use + "/"
    print("Loading pickled output", file=log.v5)
    start = time.time()
    input_folder_name = self.output_folder + "extracted/"
    if self.pre_saved_to_use in "":
      num_outputs_already_saved = len(os.listdir(input_folder_name))
      input_file_name = input_folder_name + "extracted" + str(num_outputs_already_saved - 1).zfill(4) + ".pkl"
    else:
      input_file_name = input_folder_name + self.pre_saved_to_use + ".pkl"
    with open(input_file_name, 'rb') as inputfile:
      results = pickle.load(inputfile)
    track_ys = results["track_ys"]
    track_tags_name = results["track_tags_name"]
    track_crop_ids = results["track_crop_ids"]
    track_classes = results["track_classes"]
    max_classes = results["max_classes"]
    center_ids = results["center_ids"]
    track_labels = results["track_labels"]
    # center_ids = track_labels = None
    track_ims = []
    original_track_ys = numpy.array(track_ys)
    for idx in track_crop_ids:
      crop_path = crop_folder + ("%07d" % idx) + ".jpg"
      # im = imread(crop_path)
      # track_ims.append(im)
    print("Loaded, elapsed: ", time.time() - start, file=log.v5)

    # track_ims = [imresize(t, [self.s, self.s]) for t in track_ims]

    return original_track_ys, track_ims, track_classes, track_crop_ids, track_tags_name, max_classes, center_ids,track_labels

  def combine_datasets(self):
    ######## TEMPORARY CODE FOR SAVING KITTI+KITTI+OXFORD+SCHIPOL TOGETHER ##########
    ### Load vectors and tags
    track_ims = []
    track_ys = []
    track_tags_name = []
    track_crop_ids = []
    track_classes = []
    last_crop = 0
    curr_crop = 0
    ccs = ["KITTI_RAW2", "KITTI_TRAIN", "OXFORD2", "SCHIPHOL2"]
    for cc in ccs:
      crop_folder = self.output_folder + "crops/" + cc + "/"
      print("Loading pickled output " + cc, file=log.v5)
      start = time.time()
      input_folder_name = self.output_folder + "extracted/"
      input_file_name = input_folder_name + cc + ".pkl"
      with open(input_file_name, 'rb') as inputfile:
        results = pickle.load(inputfile)
      ctrack_ys = results["track_ys"]
      ctrack_tags_name = results["track_tags_name"]
      ctrack_crop_ids = results["track_crop_ids"]
      ctrack_classes = results["track_classes"]
      for crop,(y,tag,clas) in enumerate(zip(ctrack_ys,ctrack_tags_name,ctrack_classes)):
        track_ys.append(y)
        track_tags_name.append(tag)
        track_crop_ids.append(last_crop+crop)
        track_classes.append(clas)
        curr_crop = last_crop + crop
      last_crop = curr_crop + 1
      print("Loaded, "+cc+"elapsed: ", time.time() - start, file=log.v5)
      start = time.time()
      for idx in ctrack_crop_ids:
        crop_path = crop_folder + ("%07d" % idx) + ".jpg"
        im = imread(crop_path)
        track_ims.append(im)
      print("Loaded ims "+cc+", elapsed: ", time.time() - start, file=log.v5)
    ## Downscale images
    # track_ims = [imresize(im, [self.s, self.s]) for im in track_ims]

    print("saving")
    start = time.time()
    im_path = self.output_folder + "crops/" + "COMBINED" + "/"
    tf.gfile.MakeDirs(im_path)
    for idx, im in enumerate(track_ims):
      im_name = im_path + ("%07d" % idx) + ".jpg"
      imsave(im_name, im)
    print("saved images", time.time()-start)

    ## Save extracted
    print("Saving pickled output", file=log.v5)
    start = time.time()
    results = {"track_ys": track_ys, "track_tags_name": track_tags_name, "track_crop_ids": track_crop_ids,
               "track_classes": track_classes}
    output_folder_name = self.output_folder + "extracted/"
    tf.gfile.MakeDirs(output_folder_name)
    output_file_name = output_folder_name + "COMBINED.pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)
    quit(0)

  def cluster_and_classify(self,track_ys, track_classes, xxx_todo_changeme,savedMemory=Memory(cachedir=None, verbose=0), mode=0):
    (n_components, min_cluster_size, min_samples) = xxx_todo_changeme
    from hdbscan import HDBSCAN, all_points_membership_vectors, approximate_predict
    start = time.time()
    clusterer = None

    if track_classes is not None:
      _, class_ids = numpy.unique(track_classes, return_inverse=True)

      # ## Reorder classes to be in size order
      # orig_ids, sizes = numpy.unique(class_ids, return_counts=True)
      # new_ids = numpy.argsort(sizes)[::-1]
      # mapping = dict(zip(new_ids, orig_ids))
      # new_class_ids = numpy.copy(class_ids)
      # for k, v in mapping.iteritems(): class_ids[new_class_ids == k] = v
    else:
      class_ids = None

    if mode==1:
      clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-2,
                          algorithm='boruvka_kdtree',cluster_selection_method='eom', prediction_data=True, memory=savedMemory).fit(track_ys)
      soft_clusters = all_points_membership_vectors(clusterer)
      # cluster_ids = numpy.array([numpy.argmax(x) if numpy.max(x)>1.0/len(x) else -1 for x in soft_clusters])
      cluster_ids = numpy.array([numpy.argmax(x) for x in soft_clusters])
    elif mode==2:
      cluster_ids = class_ids
    else:
      clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-2,
                          algorithm='boruvka_kdtree',
                          cluster_selection_method='eom',prediction_data=True, memory=savedMemory)
      # clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-2,
      #                     algorithm='boruvka_kdtree',
      #                     cluster_selection_method='leaf', prediction_data=True, memory=savedMemory)
      cluster_ids = clusterer.fit_predict(track_ys)

    # ## Reorder clusters to be in size order
    # orig_ids, sizes = numpy.unique(cluster_ids, return_counts=True)
    # orig_ids = orig_ids[1:]
    # sizes = sizes[1:]
    # new_ids = numpy.argsort(sizes)[::-1]
    # mapping = dict(zip(new_ids, orig_ids))
    # new_cluster_ids = numpy.copy(cluster_ids)
    # for k, v in mapping.iteritems(): cluster_ids[new_cluster_ids == k] = v

    duration = time.time() - start
    print(n_components, min_samples, min_cluster_size, duration, file=log.v5)

    return cluster_ids, class_ids,clusterer

  def cluster_all_options(self,original_track_ys):
    total_clustering_start = time.time()
    num_outputs_already_saved = len(os.listdir(self.output_folder + "tree_cache/"))
    savedMemory = Memory(self.output_folder + "tree_cache/" + str(num_outputs_already_saved).zfill(4) + "/")
    num_outputs_already_saved = len(os.listdir(self.output_folder + "tests/"))
    self.test_output_file = self.output_folder + "tests/" + str(num_outputs_already_saved).zfill(3) + "/"

    for n_components in [8, 12, 16, 24, 32, 48, 64, 96, 128]:
      start_this_dimensions = time.time()
      pca = PCA(n_components=n_components)
      track_ys = pca.fit_transform(original_track_ys)
      for min_samples in range(6, 51, 2):
        start_this_run = time.time()
        for min_cluster_size in range(6, 51, 2):
          cluster_ids, _,_ = self.cluster_and_classify(track_ys, None, (n_components, min_cluster_size, min_samples), savedMemory)
          cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(cluster_ids, track_classes)
          self.write_summary(cluster_ids, cluster_class_list, cluster_class_counts,
                             (n_components, min_cluster_size, min_samples))
        print("this run elapsed =", time.time() - start_this_run, file=log.v5)
      print("this dimensionality elapsed =", time.time() - start_this_dimensions, file=log.v5)
    print("total clustering elapsed =", time.time() - total_clustering_start, file=log.v5)

    return

  def create_cluster_class_lists(self, cluster_ids, track_classes):

    # all_class_list, all_class_counts = numpy.unique(track_classes, return_counts=True)
    #
    # # Remove unknown type
    # now_list = [(a, b) for a, b in zip(all_class_list, all_class_counts)]
    # now_dict = dict(now_list)
    # u_count = now_dict.get('unknown_type', 0) + now_dict.get('unknown', 0)
    # now_dict['unknown'] = u_count
    # now_dict.pop("unknown_type", None)
    # all_class_list = numpy.array(list(now_dict.keys()))
    # all_class_counts = numpy.array(list(now_dict.values()))
    #
    # # Sort in order
    # all_class_arg_sort_ids = numpy.argsort(all_class_counts)[::-1]
    # all_class_list = all_class_list[all_class_arg_sort_ids]
    # all_class_counts = all_class_counts[all_class_arg_sort_ids]

    cluster_class_list = []
    cluster_class_counts = []
    cluster_id_list, cluster_sizes = numpy.unique(cluster_ids, return_counts=True)
    ids = numpy.arange(len(cluster_ids))
    for ii, (cluster_id, cluster_size) in enumerate(zip(cluster_id_list, cluster_sizes)):
      curr = ids[cluster_ids == cluster_id]
      curr_classes = [track_classes[i] for i in curr]
      class_list, class_counts = numpy.unique(curr_classes, return_counts=True)

      # Remove unknown type
      now_list = [(a, b) for a, b in zip(class_list, class_counts)]
      now_dict = dict(now_list)
      u_count = now_dict.get('unknown_type', 0) + now_dict.get('unknown', 0)
      now_dict['unknown'] = u_count
      now_dict.pop("unknown_type", None)
      class_list = numpy.array(list(now_dict.keys()))
      class_counts = numpy.array(list(now_dict.values()))

      # Sort in order
      class_arg_sort_ids = numpy.argsort(class_counts)[::-1]
      class_list = class_list[class_arg_sort_ids]
      class_counts = class_counts[class_arg_sort_ids]

      cluster_class_list.append(class_list)
      cluster_class_counts.append(class_counts)

    # all_class_list = numpy.unique(sum((c for c in cluster_class_list), []))
    # all_class_counts = [sum((c for c,l in zip(cluster_class_counts,cluster_class_counts) if l==a), 0) for a in all_class_list]

    return cluster_class_list, cluster_class_counts

  def get_names_and_group_outliers(self, cluster_class_list, cluster_class_counts):
    names = []
    group_outlier_counts = []
    for i,(class_list, class_counts) in enumerate(zip(cluster_class_list,cluster_class_counts)):
      class_list_without_unknown = [l for l in class_list if l != 'unknown']
      class_counts_without_unknown = [c for l, c in zip(class_list, class_counts) if l != 'unknown']
      if i == 0:
        name = 'outliers'
        group_outliers = 0
      elif len(class_list) == 1:
        name = class_list[0]
        group_outliers = 0
      elif len(class_list_without_unknown) == 1:
        name = class_list_without_unknown[0]
        group_outliers = 0
      else:
        cidx = numpy.argmax(class_counts_without_unknown)
        main, main_count = (class_list_without_unknown[cidx], class_counts_without_unknown[cidx])
        group_outliers = sum([c for l, c in zip(class_list_without_unknown, class_counts_without_unknown) if l != main])
        if main_count > group_outliers:
          name = main
        else:
          name = 'unknown'

      names.append(name)
      group_outlier_counts.append(group_outliers)

    required_outliers = max(group_outlier_counts)
    all_names, name_inverses, name_count = numpy.unique(names, return_inverse=True, return_counts=True)
    all_ids = numpy.arange(len(name_inverses))
    extra_counts = []
    for a_idx, (a_name, a_count) in enumerate(zip(all_names, name_count)):
      if a_count > 1 and a_name not in 'unknown':
        ids = all_ids[name_inverses == a_idx]
        counts_list = cluster_class_counts[a_idx]
        counts_list = sorted(counts_list,reverse=True)
        extra_counts_list = counts_list[1:]
        extra_counts.append(sum(extra_counts_list))
    if extra_counts:
      tot_extra = max(extra_counts)
    else:
      tot_extra = 0
    if required_outliers < tot_extra:
      required_outliers = tot_extra
    # if required_outliers >= min_cluster_size:
    #   required_outliers += 1000
    for ii, (counts,list) in enumerate(zip(cluster_class_counts,cluster_class_list)):
      class_counts_without_unknown = [c for l, c in zip(list, counts) if l != 'unknown']
      if ii > 0 and len(class_counts_without_unknown)>0 and (class_counts_without_unknown[0] <= required_outliers):# or class_counts_without_unknown[0] < 0.5 * sum(counts)):
        names[ii] = 'unknown'

    all_names, name_inverses, name_count = numpy.unique(names, return_inverse=True, return_counts=True)
    for a_idx, (a_name, a_count) in enumerate(zip(all_names, name_count)):
      if a_count > 1:
        ids = all_ids[name_inverses == a_idx]
        for ii, id in enumerate(ids):
          names[id] += '_' + str(ii)

    return names, group_outlier_counts,required_outliers

  def write_summary(self, cluster_ids, cluster_class_list, cluster_class_counts, xxx_todo_changeme1):
    (n_components, min_cluster_size, min_samples) = xxx_todo_changeme1
    final_output_fol = self.test_output_file + str(n_components).zfill(3) + "_" + str(min_samples).zfill(3) + "_" + str(min_cluster_size).zfill(3) + "/"
    if not os.path.exists(final_output_fol):
      os.makedirs(final_output_fol)

    num_outliers = len(cluster_ids[cluster_ids == -1])
    num_classes = len(numpy.unique(cluster_ids)) - 1
    duration = ""

    all_class_list = numpy.unique(sum((list(c) for c in cluster_class_list),[]))
    all_class_counts = [sum(c for c, l in zip(cluster_class_counts, cluster_class_counts) if l == a) for a in all_class_list]

    ### Create summary file:
    text_file_name = final_output_fol + "summary.txt"
    F = open(text_file_name, 'w')
    F.write("Cluster method: " + "HDBSCAN" + "\n")
    F.write("n_components: " + str(n_components) + "\n")
    F.write("min_samples: " + str(min_samples) + "\n")
    F.write("min_cluster_size: " + str(min_cluster_size) + "\n")
    F.write("clustering time: " + str(duration) + "\n")
    F.write(" " + "\n")
    F.write("Num Classes: " + str(num_classes) + "\n")
    F.write("Num Outliers: " + str(num_outliers) + "\n")
    F.write(" " + "\n")

    cluster_string = "all_tracks: " + str(len(cluster_ids))
    for name, count in zip(all_class_list, all_class_counts):
      cluster_string += ", " + name + ": " + str(count)
    F.write(cluster_string + "\n")
    F.write(" " + "\n")

    cluster_id_list, cluster_sizes = numpy.unique(cluster_ids, return_counts=True)
    for cluster_id, cluster_size, class_list, class_counts in zip(cluster_id_list, cluster_sizes, cluster_class_list, cluster_class_counts):
      cluster_string = "#: " + str(cluster_id).zfill(3) + ", Size: " + str(cluster_size)
      for name, count in zip(class_list, class_counts):
        cluster_string += ", " + name + ": " + str(count)
      F.write(cluster_string + "\n")

    return

  def read_summary(self, final_output_fol):
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
    cluster_class_list = cluster_class_counts = []
    for i in range(num_classes + 1):
      strs = (F.readline().split(', ')[2:])
      cluster_class_list.append([cstr.split(': ')[0] for cstr in strs])
      cluster_class_counts.append([int(cstr.split(': ')[1]) for cstr in strs])

    return num_classes, num_outliers, cluster_class_list, cluster_class_counts

  def optimize_params(self):
    ### Optimize parameters and name clusters
    if self.cluster_all_options:
      num_outputs_already_saved = len(os.listdir(self.output_folder + "tests/"))
      test_output_file = self.output_folder + "tests/" + str(num_outputs_already_saved).zfill(3) + "/"
    else:
      test_output_file = self.output_folder + "tests/" + self.pre_saved_to_use + "/"

    start = time.time()
    results = []
    for n_components in [8, 12, 16, 24, 32, 48, 64, 96, 128]:
      for min_samples in range(6, 51, 2):
        for min_cluster_size in range(6, 51, 2):
          final_output_fol = test_output_file + str(n_components).zfill(3) + "_" + str(min_samples).zfill(
            3) + "_" + str(min_cluster_size).zfill(3) + "/"
          num_classes, num_outliers, cluster_class_list, cluster_class_counts = self.read_summary(final_output_fol)
          names, group_outlier_counts, required_outliers = self.get_names_and_group_outliers(cluster_class_list,
                                                                                             cluster_class_counts)
          results.append(
            (required_outliers, n_components, min_samples, min_cluster_size, num_classes, num_outliers, names))

    results = sorted(results, key=lambda x: x[0])
    for i in range(40):
      print(results[i])
    goods = results[0][1:4]
    print(time.time() - start)

    output_file_name = self.output_folder + "goods.pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(goods, outputfile)

  def output_images(self, track_ims, cluster_ids, track_classes, cluster_class_list, cluster_class_counts,final_output_fol, track_tags = None, mode = 0):
    ### Save images for each cluster
    num_row = 20
    calc_time = 0
    save_time = 0
    start = time.time()

    mode = 1

    if mode == 1:

      # Save images for each tracks
      print("saving clusters", file=log.v5)
      start = time.time()
      for idx, cluster_id in enumerate(cluster_ids):
        curr_fol = final_output_fol + str(cluster_id).zfill(3) + "/"
        tf.gfile.MakeDirs(curr_fol)
        # new_path = curr_fol + str(idx).zfill(5) + '.jpg'
        yo_name = track_tags[idx].split('/')[-1]
        new_path = curr_fol + yo_name + '.jpg'
        im = track_ims[idx]
        imsave(new_path, im)
      print("saving elapsed =", time.time() - start, file=log.v5)

    else:

      for curr_id, (class_list, class_count) in enumerate(zip(cluster_class_list, cluster_class_counts)):
        curr_id -= 1
        print(len(track_ims))
        print(len(cluster_ids))
        curr_ims = [im for im,c in zip(track_ims,cluster_ids) if curr_id == c]
        curr_class = [c if c!='unknown_type' else 'unknown' for c,n in zip(track_classes,cluster_ids) if curr_id == n]
        cluster_size = len(curr_ims)
        start_calc = time.time()

        # if mode==1:

          # ## Save images for each track
          # num_col = int(numpy.floor(len(curr_ims)/num_row))+1
          # whole_im = numpy.zeros([self.s*num_col,self.s*num_row,3])
          # for id,im in enumerate(curr_ims):
          #   x = int(numpy.floor(id / num_row) * self.s)
          #   y = (id % num_row)*self.s
          #   whole_im[x:x+self.s, y:y+self.s, :] = im
          # curr_file = final_output_fol + str(curr_id).zfill(3) + '---' + str(cluster_size) + ".jpg"
          # calc_time += time.time()-start_calc
          # start_save = time.time()
          # imsave(curr_file, whole_im)
          # save_time += time.time()-start_save

        # else:
        num_cols = [int(numpy.floor(cc / num_row)) + 2 for cc in class_count]
        y_start = 0
        y_starts = []
        for c in num_cols:
          y_starts.append(y_start)
          y_start = y_start + c
        whole_im = numpy.zeros([self.s * sum(num_cols), self.s * num_row, 3])
        for class_id, (name, count, num_col, y_start) in enumerate(zip(class_list, class_count, num_cols, y_starts)):

          test = [i for i in curr_class if i == name]
          print(len(test))
          print(len(curr_ims))

          these_ims = [curr_ims[ii] for ii, i in enumerate(curr_class) if i == name]
          for id, im in enumerate(these_ims):
            x = y_start * self.s + int(numpy.floor(id / num_row) * self.s)
            y = (id % num_row) * self.s
            whole_im[x:x + self.s, y:y + self.s, :] = im
        curr_file = final_output_fol + str(curr_id).zfill(3) + '---' + str(cluster_size) + ".jpg"
        calc_time += time.time() - start_calc
        start_save = time.time()
        try:
          imsave(curr_file, whole_im)
        except:
          print("failed")
        save_time += time.time() - start_save
    duration = time.time() - start
    print("saving elapsed =", duration, calc_time, save_time, file=log.v5)
    return

  def save_for_retraining(self, track_ys, track_tags_name, cluster_ids, names,output_file=None):
    ## Saving file of tracks to cluster labels
    print("Saving pickled cluster assignment output", file=log.v5)
    start = time.time()
    results = {"avg_ys": track_ys, "tags": track_tags_name, "cluster_label": cluster_ids}
    print(names)
    results["class_labels"] = numpy.array(names, dtype="string")
    output_folder_name = self.output_folder + "assigned_output/"
    tf.gfile.MakeDirs(output_folder_name)
    if output_file is None:
      # num_outputs_already_saved = len(os.listdir(output_folder_name))
      # output_file_name = output_folder_name + "labels" + str(num_outputs_already_saved).zfill(4) + ".pkl"
      output_file_name = output_folder_name + self.exp_name + ".pkl"
    else:
      fol = output_file.split('/')[0]
      tf.gfile.MakeDirs(output_folder_name+fol)
      output_file_name = output_folder_name + output_file + ".pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)
    return

  def save_clusterer_info(self, clusterer, pca):
    start = time.time()
    results = {"clusterer": clusterer, "pca": pca}
    output_folder_name = self.output_folder + "clusterer_running_data/"
    tf.gfile.MakeDirs(output_folder_name)
    # num_outputs_already_saved = len(os.listdir(output_folder_name))
    # output_file_name = output_folder_name + "labels" + str(num_outputs_already_saved).zfill(4) + ".pkl"
    output_file_name = output_folder_name + self.exp_name + ".pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)
    return

  def read_clustering_info(self):
    print("Loading pickled output", file=log.v5)
    start = time.time()
    input_file_name = self.output_folder + "clusterer_running_data/" + self.clusterer_to_use + ".pkl"
    with open(input_file_name, 'rb') as inputfile:
      results = pickle.load(inputfile)
    clusterer = results["clusterer"]
    pca = results["pca"]
    print("Loaded, elapsed: ", time.time() - start, file=log.v5)
    return clusterer, pca

  def run_one_whole_clustering(self, original_track_ys, track_classes, track_ims, track_tags_name, ys = None, tags=None):
    from hdbscan import HDBSCAN, all_points_membership_vectors, approximate_predict
    if self.clustering_meta_params:
      n_components, min_samples, min_cluster_size = self.clustering_meta_params
    else:
      input_file_name = self.output_folder + "goods.pkl"
      with open(input_file_name, 'rb') as inputfile:
        to_use = pickle.load(inputfile)
      n_components, min_samples, min_cluster_size = to_use

    # num_outputs_already_saved = len(os.listdir(self.output_folder + "tests/"))
    # self.test_output_file = self.output_folder + "tests/" + str(num_outputs_already_saved).zfill(3) + "/"
    self.test_output_file = self.output_folder + "tests/" + self.exp_name + "/"
    pca = PCA(n_components=n_components)
    track_ys = pca.fit_transform(original_track_ys)
    cluster_ids, class_ids, clusterer = self.cluster_and_classify(track_ys, track_classes, (n_components, min_cluster_size, min_samples))
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(cluster_ids, track_classes)
    names, group_outlier_counts, required_outliers = self.get_names_and_group_outliers(cluster_class_list,cluster_class_counts)

    # ## Combining clustering and detection labels
    # track_classes = numpy.array(track_classes)
    # curr_det_labels = track_classes[cluster_ids == -1]
    # all_cluster_ids = numpy.arange(len(track_classes))
    # curr_cluster_ids = all_cluster_ids[cluster_ids==-1]
    # for c,id in zip(curr_det_labels,curr_cluster_ids):
    #   if c != 'unknown':
    #     for ii,n in enumerate(names):
    #       if c==n:
    #         cluster_ids[id] = ii-1
    # cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(cluster_ids, track_classes)
    # names, group_outlier_counts, required_outliers = self.get_names_and_group_outliers(cluster_class_list,cluster_class_counts)

    self.write_summary(cluster_ids, cluster_class_list, cluster_class_counts,(n_components, min_cluster_size, min_samples))
    final_output_fol = self.test_output_file + str(n_components).zfill(3) + "_" + str(min_samples).zfill(3) + "_" + str(min_cluster_size).zfill(3) + "/"
    # self.output_images(track_ims, cluster_ids, track_classes, cluster_class_list, cluster_class_counts,final_output_fol)
    self.output_images( track_ims, cluster_ids, track_classes, cluster_class_list, cluster_class_counts, final_output_fol,track_tags_name,mode =1)
    self.save_for_retraining(track_ys, track_tags_name, cluster_ids, names)
    print("saved images")
    self.save_clusterer_info(clusterer,pca)
    print("saved cluster")

    # ## Run forwarder on all images
    # ys = numpy.array(ys)
    # t_ys = pca.transform(ys)
    # new_cluster_ids, strengths = approximate_predict(clusterer, t_ys)
    # new_track_classes = [tag.split("___")[-1] for tag in tags]
    # new_cluster_class_list, new_cluster_class_counts = self.create_cluster_class_lists(new_cluster_ids,new_track_classes)
    # self.test_output_file = self.output_folder + "tests/" + str(num_outputs_already_saved).zfill(3) + "/full/"
    # self.write_summary(new_cluster_ids, new_cluster_class_list, new_cluster_class_counts,(n_components, min_cluster_size, min_samples))
    # new_names, new_group_outlier_counts, new_required_outliers = self.get_names_and_group_outliers(new_cluster_class_list,new_cluster_class_counts)
    # new_final_output_fol = self.test_output_file + str(n_components).zfill(3) + "_" + str(min_samples).zfill(3) + "_" + str(min_cluster_size).zfill(3) + "/"
    # new_track_ims = [imread(self.crop_folder + ("%07d" % idx) + ".jpg") for idx in range(len(new_cluster_ids))]
    # self.output_images(new_track_ims, new_cluster_ids, new_track_classes, new_cluster_class_list, new_cluster_class_counts,new_final_output_fol)
    #
    # ## Quick analysis of tracks:
    # track_tags_name, track_idx = numpy.unique(["---".join(tag.split("___")[:-2]) for tag in tags], return_inverse=True)
    # track_tags = numpy.unique(track_idx)
    # wrong = 0
    # right = 0
    # for track_tag in track_tags:
    #   iidx = numpy.arange(len(track_idx))
    #   curr_ids = iidx[track_idx == track_tag]
    #   # crop_path = self.crop_folder + ("%07d" % idx) + ".jpg"
    #   # im = imread(crop_path)
    #   curr_class_labels = [new_track_classes[x] for x in curr_ids]
    #   curr_cluster_labels = [new_cluster_ids[x] for x in curr_ids]
    #   if len(set(curr_cluster_labels)) > 1:
    #     print curr_cluster_labels
    #     print curr_class_labels
    #     print ''
    #     wrong+=1
    #   else:
    #     right+=1
    # print right, wrong
    # self.save_for_retraining(t_ys, tags, new_cluster_ids, new_names)

    # # Recluster outliers
    # track_classes = numpy.array(track_classes)
    # track_ims = numpy.array(track_ims)
    # new_track_ys = track_ys[cluster_ids==-1]
    # new_track_classes = track_classes[cluster_ids==-1]
    # new_track_ims = track_ims[cluster_ids == -1]
    # new_cluster_ids, new_class_ids, _ = self.cluster_and_classify(new_track_ys, new_track_classes,(n_components, min_cluster_size, min_samples))
    # new_cluster_class_list, new_cluster_class_counts = self.create_cluster_class_lists(new_cluster_ids, new_track_classes)
    # self.test_output_file = self.output_folder + "tests/" + str(num_outputs_already_saved).zfill(3) + "/resclustering/"
    # self.write_summary(new_cluster_ids, new_cluster_class_list, new_cluster_class_counts,(n_components, min_cluster_size, min_samples))
    # # new_names, new_group_outlier_counts, new_required_outliers = self.get_names_and_group_outliers(new_cluster_class_list,new_cluster_class_counts)
    # new_final_output_fol = self.test_output_file + str(n_components).zfill(3) + "_" + str(min_samples).zfill(3) + "_" + str(min_cluster_size).zfill(3) + "/"
    # self.output_images(new_track_ims, new_cluster_ids, new_track_classes, new_cluster_class_list, new_cluster_class_counts,new_final_output_fol)

    return

  def run_all_forwarding_experiements(self,original_ys,tags,original_track_ys,track_tags_name,track_classes, max_classes):
    from hdbscan import approximate_predict
    clusterer,pca = self.read_clustering_info()
    # original_ys, tags, crop_ids = self.get_all_latents_from_file()
    # original_track_ys, track_ims, track_classes, track_crop_ids, track_tags_name,max_classes = self.extract_centroids(original_ys,tags)
    classes = [tag.split("___")[-1] for tag in tags]
    original_ys = numpy.array(original_ys)
    ys = pca.transform(original_ys)
    track_ys = pca.transform(original_track_ys)

    #1a - Det Ind
    _, curr_ids = numpy.unique(classes, return_inverse=True)
    curr_ids = curr_ids-1
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids,classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name =  "1a-DetInd"
    self.save_for_retraining(ys, tags, curr_ids, names,output_file=self.exp_name+"/" + test_name)
    self.run_eval(test_name)

    #1b - Det cntr
    _, curr_ids = numpy.unique(track_classes, return_inverse=True)
    curr_ids = curr_ids - 1
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids,track_classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = "1b-DetCtr"
    self.save_for_retraining(track_ys, track_tags_name, curr_ids, names,output_file=self.exp_name+"/"+test_name)
    self.run_eval(test_name)

    # 1c - Det max
    _, curr_ids = numpy.unique(max_classes, return_inverse=True)
    curr_ids = curr_ids - 1
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids, max_classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = "1c-DetMax"
    self.save_for_retraining(track_ys, track_tags_name, curr_ids, names,output_file=self.exp_name+"/" + test_name)
    self.run_eval(test_name)

    #2a - Clus Ind
    curr_ids, strengths = approximate_predict(clusterer, ys)
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids, classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = "2a-ClusInd"
    self.save_for_retraining(ys, tags, curr_ids, names,output_file=self.exp_name+"/"+test_name)
    self.run_eval(test_name)

    #2b - Clus cntr
    curr_ids, strengths = approximate_predict(clusterer, track_ys)
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids, track_classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = "2b-ClusCtr"
    self.save_for_retraining(track_ys, track_tags_name, curr_ids, names,output_file=self.exp_name+"/" + test_name)
    self.run_eval(test_name)

    # 2c - Clus max
    all_ids, strengths = approximate_predict(clusterer, ys)
    _, track_idx = numpy.unique(["---".join(tag.split("___")[:-2]) for tag in tags], return_inverse=True)
    unique_track_ids = numpy.unique(track_idx)
    max_ids = []
    for unique_track_id in unique_track_ids:
      iidx = numpy.arange(len(track_idx))
      curr_ids = iidx[track_idx == unique_track_id]
      curr_track_ids = [all_ids[x] for x in curr_ids]
      # New Max
      unique_curr_ids, unique_curr_ids_counts = numpy.unique(curr_track_ids,return_counts=True)
      max_id = numpy.argmax(unique_curr_ids_counts)
      max_curr_id = unique_curr_ids[max_id]
      # # OLD MAXX
      # curr_track_ids_no_outliers = [x for x in curr_track_ids if x != -1]
      # if len(curr_track_ids_no_outliers) < 1:
      #   max_curr_id = -1
      # else:
      #   unique_curr_ids, unique_curr_ids_counts = numpy.unique(curr_track_ids_no_outliers,return_counts=True)
      #   if len(unique_curr_ids) == 1:
      #     max_curr_id = unique_curr_ids[0]
      #   else:
      #     print "WARNING, track has multiple hard clusters", [(l, c) for l, c in zip(unique_curr_ids, unique_curr_ids_counts)]
      #     max_id = numpy.argmax(unique_curr_ids_counts)
      #     max_curr_id = unique_curr_ids[max_id]
      max_ids.append(max_curr_id)
    curr_ids = max_ids
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids, max_classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = "2c-ClusMax"
    self.save_for_retraining(track_ys, track_tags_name, curr_ids, names, output_file=self.exp_name + "/"+test_name)
    self.run_eval(test_name)

    # 3a - Comb Ind
    curr_ids, strengths = approximate_predict(clusterer, ys)
    classes = numpy.array(classes)
    curr_det_labels = classes[curr_ids == -1]
    all_cluster_ids = numpy.arange(len(classes))
    curr_cluster_ids = all_cluster_ids[curr_ids==-1]
    for c,id in zip(curr_det_labels,curr_cluster_ids):
      if c != 'unknown':
        for ii,n in enumerate(names):
          if c==n:
            curr_ids[id] = ii-1
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids, classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = "3a-CombInd"
    self.save_for_retraining(ys, tags, curr_ids, names,output_file=self.exp_name+"/"+test_name)
    self.run_eval(test_name)

    # 3b - Comb Ctr
    curr_ids, strengths = approximate_predict(clusterer, track_ys)
    track_classes = numpy.array(track_classes)
    curr_det_labels = track_classes[curr_ids == -1]
    all_cluster_ids = numpy.arange(len(track_classes))
    curr_cluster_ids = all_cluster_ids[curr_ids == -1]
    for c, id in zip(curr_det_labels, curr_cluster_ids):
      if c != 'unknown':
        for ii, n in enumerate(names):
          if c == n:
            curr_ids[id] = ii - 1
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids, track_classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = "3b-CombCtr"
    self.save_for_retraining(track_ys, track_tags_name, curr_ids, names, output_file=self.exp_name + "/" + test_name)
    self.run_eval(test_name)

    # 3c - Comb Ctr
    curr_ids = max_ids
    max_classes = numpy.array(max_classes)
    curr_det_labels = max_classes[curr_ids == -1]
    all_cluster_ids = numpy.arange(len(max_classes))
    curr_cluster_ids = all_cluster_ids[curr_ids == -1]
    for c, id in zip(curr_det_labels, curr_cluster_ids):
      if c != 'unknown':
        for ii, n in enumerate(names):
          if c == n:
            curr_ids[id] = ii - 1
    cluster_class_list, cluster_class_counts = self.create_cluster_class_lists(curr_ids, max_classes)
    names, _, _ = self.get_names_and_group_outliers(cluster_class_list, cluster_class_counts)
    test_name = '3c-CombMax'
    self.save_for_retraining(track_ys, track_tags_name, curr_ids, names, output_file=self.exp_name + "/" + test_name)
    self.run_eval(test_name)

    return

  def run_eval(self,test_name):
    print("running eval", test_name)
    start = time.time()
    curr_type = self.pre_saved_to_use.split('-')[-1]
    print(self.exp_name,test_name,curr_type)
    command = "/home/luiten/vision/savitar/scripts/tracklets/remap_and_eval_tracking_KITTI_train.sh /home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/assigned_output/%s/%s.pkl 2017_11_06_1444_54_mining %s_RCNN_savitar | grep 'MODA\|False Positives\|Missed Targets' | awk '{print $NF}' | sed ':a;N;$!ba;s/\\n/\\t/g'"%(self.exp_name,test_name,curr_type)
    var = os.popen(command).read()
    save_file_name = self.output_folder + "runs/" + self.exp_name + ".txt"
    with open(save_file_name,'a') as fn:
      fn.write(test_name + "\t" + var)
    print("finished eval", test_name, time.time()-start)
    return var

  def load_a_data(self,data_name):
    input_file_name = self.output_folder + data_name+ ".pkl"
    with open(input_file_name, 'rb') as inputfile:
      result = pickle.load(inputfile)
    return result

  def show_plots(self):
    import matplotlib

    matplotlib.rcParams.update({'font.size': 12})
    matplotlib.rcParams['text.usetex'] = True
    # plt.grid(True)
    lw = 4
    ms = 12.0

    results_trip = self.load_a_data("plot_results_trip_hom")
    res_extra_trip = self.load_a_data("extra_plot_res_trip_hom")
    results_RCNN = self.load_a_data("plot_results_RCNN_hom_new")
    res_extra_RCNN = self.load_a_data("extra_plot_res_RCNN_hom_new")
    # clusternet_ami,clusternet_num_classes = self.load_a_data("ClusterNetCOCOMargin4")
    results_cluster = self.load_a_data("ClusterNetCOCOMargin4extended_hom")


    import matplotlib.pyplot as plt

    result_to_show_trip = numpy.array(results_trip)
    plt.plot(result_to_show_trip[:, 0], result_to_show_trip[:, 1], 'b-', linewidth=lw, markersize=ms)
    plt.plot([0, 0.5], [res_extra_trip["kmeans"], res_extra_trip["kmeans"]], 'b--', linewidth=lw, markersize=ms)
    # plt.plot(res_extra_trip["gmm-tied"], 'bs')
    # plt.plot(res_extra_trip["gmm-diag"], 'bo')
    # plt.plot(res_extra_trip["gmm-spherical"], 'b*')
    plt.plot([0, 0.5], [res_extra_trip["gmm-full"], res_extra_trip["gmm-full"]], 'b:', linewidth=lw, markersize=ms)

    result_to_show_RCNN = numpy.array(results_RCNN)
    plt.plot(result_to_show_RCNN[:, 0], result_to_show_RCNN[:, 1], 'r-', linewidth=lw, markersize=ms)
    plt.plot([0, 0.5], [res_extra_RCNN["kmeans"], res_extra_RCNN["kmeans"]], 'r--', linewidth=lw, markersize=ms)
    # plt.plot(res_extra_RCNN["gmm-tied"], 'rs')
    # plt.plot(res_extra_RCNN["gmm-diag"], 'ro')
    # plt.plot(res_extra_RCNN["gmm-spherical"], 'r*')
    plt.plot([0, 0.5], [res_extra_RCNN["gmm-full"], res_extra_RCNN["gmm-full"]], 'r:', linewidth=lw, markersize=ms)

    result_to_show_cluster = numpy.array(results_cluster)
    plt.plot(result_to_show_cluster[:, 0], result_to_show_cluster[:, 1], 'k-', linewidth=lw, markersize=ms)

    # plt.plot(clusternet_ami,'kD')

    plt.plot(result_to_show_cluster[-1, 0], result_to_show_cluster[-1, 1], 'ko', linewidth=lw, markersize=ms)
    plt.plot(result_to_show_trip[0, 0], result_to_show_trip[0, 1], 'bo', linewidth=lw, markersize=ms)
    plt.plot(res_extra_trip["kmeans"], 'bx', linewidth=lw, markersize=ms)
    plt.plot(res_extra_trip["gmm-full"], 'bx', linewidth=lw, markersize=ms)
    plt.plot(result_to_show_RCNN[0, 0], result_to_show_RCNN[0, 1], 'ro', linewidth=lw, markersize=ms)
    plt.plot(res_extra_RCNN["kmeans"], 'rx', linewidth=lw, markersize=ms)
    plt.plot(res_extra_RCNN["gmm-full"], 'rx', linewidth=lw, markersize=ms)

    ax = plt.gca()
    ax.grid(True)

    # plt.legend(("Triplet-HDBSCAN","Triplet-KMeans","Triplet-GMM-Tied","Triplet-GMM-Diag","Triplet-GMM-Spherical","Triplet-GMM-Full",
    #             "RCNN-HDBSCAN","RCNN-KMeans","RCNN-GMM-Tied","RCNN-GMM-Diag","RCNN-GMM-Spherical","RCNN-GMM-Full",
    #             "ClusterNet"))#, "Also ClusterNet"))
    plt.legend(("Triplet-HDBSCAN", "Triplet-KMeans", "Triplet-GMM",
                "RCNN-HDBSCAN", "RCNN-KMeans", "RCNN-GMM",
                "ClusterNet"))  # , "Also ClusterNet"))
    plt.xlabel("Outlier percentage")
    plt.ylabel("Homogeneity score")
    plt.title("Clustering Homogeneity results on COCO")
    plt.show()

    ######################################

    results_trip = self.load_a_data("plot_results_trip_comp")
    res_extra_trip = self.load_a_data("extra_plot_res_trip_comp")
    results_RCNN = self.load_a_data("plot_results_RCNN_comp_new")
    res_extra_RCNN = self.load_a_data("extra_plot_res_RCNN_comp_new")
    # clusternet_ami,clusternet_num_classes = self.load_a_data("ClusterNetCOCOMargin4")
    results_cluster = self.load_a_data("ClusterNetCOCOMargin4extended_comp")


    result_to_show_trip = numpy.array(results_trip)
    plt.plot(result_to_show_trip[:, 0], result_to_show_trip[:, 1], 'b-',linewidth=lw,markersize=ms)
    plt.plot([0,0.5],[res_extra_trip["kmeans"],res_extra_trip["kmeans"]],'b--',linewidth=lw,markersize=ms)
    # plt.plot(res_extra_trip["gmm-tied"], 'bs')
    # plt.plot(res_extra_trip["gmm-diag"], 'bo')
    # plt.plot(res_extra_trip["gmm-spherical"], 'b*')
    plt.plot([0,0.5],[res_extra_trip["gmm-full"],res_extra_trip["gmm-full"]], 'b:',linewidth=lw,markersize=ms)

    result_to_show_RCNN = numpy.array(results_RCNN)
    plt.plot(result_to_show_RCNN[:, 0], result_to_show_RCNN[:, 1], 'r-',linewidth=lw,markersize=ms)
    plt.plot([0,0.5],[res_extra_RCNN["kmeans"],res_extra_RCNN["kmeans"]],'r--',linewidth=lw,markersize=ms)
    # plt.plot(res_extra_RCNN["gmm-tied"], 'rs')
    # plt.plot(res_extra_RCNN["gmm-diag"], 'ro')
    # plt.plot(res_extra_RCNN["gmm-spherical"], 'r*')
    plt.plot([0,0.5],[res_extra_RCNN["gmm-full"],res_extra_RCNN["gmm-full"]], 'r:',linewidth=lw,markersize=ms)

    result_to_show_cluster = numpy.array(results_cluster)
    plt.plot(result_to_show_cluster[:, 0], result_to_show_cluster[:, 1], 'k-',linewidth=lw,markersize=ms)

    # plt.plot(clusternet_ami,'kD')

    plt.plot(result_to_show_cluster[-1, 0], result_to_show_cluster[-1, 1], 'ko',linewidth=lw,markersize=ms)
    plt.plot(result_to_show_trip[0, 0], result_to_show_trip[0, 1], 'bo',linewidth=lw,markersize=ms)
    plt.plot(res_extra_trip["kmeans"], 'bx',linewidth=lw,markersize=ms)
    plt.plot(res_extra_trip["gmm-full"], 'bx',linewidth=lw,markersize=ms)
    plt.plot(result_to_show_RCNN[0, 0], result_to_show_RCNN[0, 1], 'ro',linewidth=lw,markersize=ms)
    plt.plot(res_extra_RCNN["kmeans"], 'rx',linewidth=lw,markersize=ms)
    plt.plot(res_extra_RCNN["gmm-full"], 'rx',linewidth=lw,markersize=ms)

    ax = plt.gca()
    ax.grid(True)

    # plt.legend(("Triplet-HDBSCAN","Triplet-KMeans","Triplet-GMM-Tied","Triplet-GMM-Diag","Triplet-GMM-Spherical","Triplet-GMM-Full",
    #             "RCNN-HDBSCAN","RCNN-KMeans","RCNN-GMM-Tied","RCNN-GMM-Diag","RCNN-GMM-Spherical","RCNN-GMM-Full",
    #             "ClusterNet"))#, "Also ClusterNet"))
    plt.legend(("Triplet-HDBSCAN", "Triplet-KMeans", "Triplet-GMM",
                "RCNN-HDBSCAN", "RCNN-KMeans", "RCNN-GMM",
                "ClusterNet"))  # , "Also ClusterNet"))
    plt.xlabel("Outlier percentage")
    plt.ylabel("Completeness score")
    plt.title("Clustering Completeness results on COCO")
    plt.show()

    ######################################

    results_trip_kitti = self.load_a_data("plot_results_KITTI_TRIPLET_hom")
    res_extra_trip_kitti = self.load_a_data("extra_plot_res_KITTI_hom")
    results_RCNN = self.load_a_data("plot_results_KITTI_RCNN_hom")
    res_extra_RCNN = self.load_a_data("extra_plot_res_KITTI_RCNN_hom")
    results_cluster = self.load_a_data("ClusterNetKITTIMargin4extended_hom")

    result_to_show_kitti = numpy.array(results_trip_kitti)
    import matplotlib.pyplot as plt
    plt.plot(result_to_show_kitti[:, 0], result_to_show_kitti[:, 1], 'b-',linewidth=lw,markersize=ms)
    plt.plot([0,0.5],[res_extra_trip_kitti["kmeans"],res_extra_trip_kitti["kmeans"]], 'b--',linewidth=lw,markersize=ms)
    # plt.plot(res_extra_trip_kitti["gmm-tied"], 'bs')
    # plt.plot(res_extra_trip_kitti["gmm-diag"], 'bo')
    # plt.plot(res_extra_trip_kitti["gmm-spherical"], 'b*')
    plt.plot([0,0.5],[res_extra_trip_kitti["gmm-full"],res_extra_trip_kitti["gmm-full"]], 'b:',linewidth=lw,markersize=ms)

    result_to_show_RCNN = numpy.array(results_RCNN)
    plt.plot(result_to_show_RCNN[:, 0], result_to_show_RCNN[:, 1], 'r-',linewidth=lw,markersize=ms)
    plt.plot([0, 0.5], [res_extra_RCNN["kmeans"], res_extra_RCNN["kmeans"]], 'r--',linewidth=lw,markersize=ms)
    plt.plot([0, 0.5], [res_extra_RCNN["gmm-full"], res_extra_RCNN["gmm-full"]], 'r:',linewidth=lw,markersize=ms)

    result_to_show_cluster = numpy.array(results_cluster)
    plt.plot(result_to_show_cluster[:, 0], result_to_show_cluster[:, 1], 'k-',linewidth=lw,markersize=ms)

    plt.plot(result_to_show_kitti[0, 0], result_to_show_kitti[0, 1], 'bo',linewidth=lw,markersize=ms)
    plt.plot(res_extra_trip_kitti["kmeans"], 'bx',linewidth=lw,markersize=ms)
    plt.plot(res_extra_trip_kitti["gmm-full"], 'bx',linewidth=lw,markersize=ms)
    plt.plot(result_to_show_RCNN[0, 0], result_to_show_RCNN[0, 1], 'ro',linewidth=lw,markersize=ms)
    plt.plot(res_extra_RCNN["kmeans"], 'rx',linewidth=lw,markersize=ms)
    plt.plot(res_extra_RCNN["gmm-full"], 'rx',linewidth=lw,markersize=ms)


    plt.plot(result_to_show_cluster[-1, 0], result_to_show_cluster[-1, 1], 'ko',linewidth=lw,markersize=ms)

    ax = plt.gca()
    ax.grid(True)

    # plt.legend(("Triplet-HDBSCAN", "Triplet-KMeans", "Triplet-GMM-Tied", "Triplet-GMM-Diag", "Triplet-GMM-Spherical",
    #             "Triplet-GMM-Full"))
    plt.legend(("Triplet-HDBSCAN", "Triplet-KMeans","Triplet-GMM",
                "RCNN-HDBSCAN", "RCNN-KMeans", "RCNN-GMM",
                "ClusterNet"
                ))
    plt.xlabel("Outlier percentage")
    plt.ylabel("Homogeneity score")
    plt.title("Clustering Homogeneity results on KITTI-Raw")
    plt.show()

    ######################################

    results_trip_kitti = self.load_a_data("plot_results_KITTI_TRIPLET_comp")
    res_extra_trip_kitti = self.load_a_data("extra_plot_res_KITTI_comp")
    results_RCNN = self.load_a_data("plot_results_KITTI_RCNN_comp")
    res_extra_RCNN = self.load_a_data("extra_plot_res_KITTI_RCNN_comp")
    results_cluster = self.load_a_data("ClusterNetKITTIMargin4extended_comp")

    result_to_show_kitti = numpy.array(results_trip_kitti)
    import matplotlib.pyplot as plt
    plt.plot(result_to_show_kitti[:, 0], result_to_show_kitti[:, 1], 'b-', linewidth=lw, markersize=ms)
    plt.plot([0, 0.5], [res_extra_trip_kitti["kmeans"], res_extra_trip_kitti["kmeans"]], 'b--', linewidth=lw,
             markersize=ms)
    # plt.plot(res_extra_trip_kitti["gmm-tied"], 'bs')
    # plt.plot(res_extra_trip_kitti["gmm-diag"], 'bo')
    # plt.plot(res_extra_trip_kitti["gmm-spherical"], 'b*')
    plt.plot([0, 0.5], [res_extra_trip_kitti["gmm-full"], res_extra_trip_kitti["gmm-full"]], 'b:', linewidth=lw,
             markersize=ms)

    result_to_show_RCNN = numpy.array(results_RCNN)
    plt.plot(result_to_show_RCNN[:, 0], result_to_show_RCNN[:, 1], 'r-',linewidth=lw,markersize=ms)
    plt.plot([0, 0.5], [res_extra_RCNN["kmeans"], res_extra_RCNN["kmeans"]], 'r--',linewidth=lw,markersize=ms)
    plt.plot([0, 0.5], [res_extra_RCNN["gmm-full"], res_extra_RCNN["gmm-full"]], 'r:',linewidth=lw,markersize=ms)

    result_to_show_cluster = numpy.array(results_cluster)
    plt.plot(result_to_show_cluster[:, 0], result_to_show_cluster[:, 1], 'k-',linewidth=lw,markersize=ms)

    plt.plot(result_to_show_kitti[0, 0], result_to_show_kitti[0, 1], 'bo', linewidth=lw, markersize=ms)
    plt.plot(res_extra_trip_kitti["kmeans"], 'bx', linewidth=lw, markersize=ms)
    plt.plot(res_extra_trip_kitti["gmm-full"], 'bx', linewidth=lw, markersize=ms)
    plt.plot(result_to_show_RCNN[0, 0], result_to_show_RCNN[0, 1], 'ro',linewidth=lw,markersize=ms)
    plt.plot(res_extra_RCNN["kmeans"], 'rx',linewidth=lw,markersize=ms)
    plt.plot(res_extra_RCNN["gmm-full"], 'rx',linewidth=lw,markersize=ms)


    plt.plot(result_to_show_cluster[-1, 0], result_to_show_cluster[-1, 1], 'ko',linewidth=lw,markersize=ms)

    ax = plt.gca()
    ax.grid(True)

    # plt.legend(("Triplet-HDBSCAN", "Triplet-KMeans", "Triplet-GMM-Tied", "Triplet-GMM-Diag", "Triplet-GMM-Spherical",
    #             "Triplet-GMM-Full"))
    plt.legend(("Triplet-HDBSCAN", "Triplet-KMeans", "Triplet-GMM",
                "RCNN-HDBSCAN", "RCNN-KMeans", "RCNN-GMM",
                "ClusterNet"
                ))
    plt.xlabel("Outlier percentage")
    plt.ylabel("Completeness score")
    plt.title("Clustering Completeness results on KITTI-Raw")
    plt.show()

  def cluster_net_extension(self,ys,tags,labels):
    from hdbscan import HDBSCAN,all_points_membership_vectors, membership_vector
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score
    ys = numpy.array(ys)
    orig_ys = ys.astype('float64')
    ys = orig_ys
    labels = numpy.array(labels)

    start = time.time()
    cluster_res = []
    cluster_res1 = []
    cluster_res2 = []
    labels_pred = numpy.argmax(ys,-1)
    print("argmax", time.time() - start)
    start = time.time()
    ################ LINE BELOW IS THE PROBLEM ###############
    list_ys = ys.tolist()
    labels_score = [a[b] for a,b in zip(list_ys,labels_pred)]
    labels_score = numpy.array(labels_score)
    # labels_score = ys[:,labels_pred]
    print("score", time.time() - start)
    start = time.time()
    num_classes = len(numpy.unique(labels_pred))
    tot_num = float(len(labels_pred))
    print(labels_score.shape)
    sort_idx = numpy.argsort(labels_score)
    print("argsort", time.time() - start)
    start = time.time()
    print(sort_idx.shape)
    sort_idx = sort_idx[0:int(round(tot_num / 2))]
    print(sort_idx.shape)
    num_points_adj = len(labels[labels != -1])

    start = time.time()
    print("starting")

    curr_labels_pred = numpy.delete(labels_pred,sort_idx)
    curr_labels_true = numpy.delete(labels,sort_idx)

    print(curr_labels_pred.shape, curr_labels_true.shape, labels_pred.shape,labels.shape)
    print("deleted", time.time() - start)

    start = time.time()

    ########################################################## Adjusted ################################
    curr_labels_pred_adj = curr_labels_pred[curr_labels_true!=-1]
    curr_labels_true_adj = curr_labels_true[curr_labels_true!=-1]

    num_in = len(curr_labels_pred_adj)
    perc_out = 1 - (float(num_in) / float(num_points_adj))
    ami = adjusted_mutual_info_score(curr_labels_true_adj, curr_labels_pred_adj)
    hom = homogeneity_score(curr_labels_true_adj, curr_labels_pred_adj)
    comp = completeness_score(curr_labels_true_adj, curr_labels_pred_adj)
    cluster_res1.append((perc_out, hom))
    cluster_res2.append((perc_out, comp))

    # ami = adjusted_mutual_info_score(curr_labels_true, curr_labels_pred)
    # num_in = float(len(curr_labels_pred))
    # perc_out = 1 - (num_in/tot_num)
    print(perc_out,ami)
    cluster_res.append((perc_out,ami))

    print("finished 1", time.time()-start)

    for cc, id in enumerate(sort_idx[::-1]):
      curr_labels_pred = numpy.concatenate((curr_labels_pred,numpy.array([labels_pred[id],])))
      curr_labels_true = numpy.concatenate((curr_labels_true,numpy.array([labels[id],])))
      if cc%80 ==79:
        # print "concating", time.time() - start
        start = time.time()

        ########################################################## Adjusted ################################
        curr_labels_pred_adj = curr_labels_pred[curr_labels_true != -1]
        curr_labels_true_adj = curr_labels_true[curr_labels_true != -1]

        num_in = len(curr_labels_pred_adj)
        perc_out = 1 - (float(num_in) / float(num_points_adj))
        ami = adjusted_mutual_info_score(curr_labels_true_adj, curr_labels_pred_adj)
        hom = homogeneity_score(curr_labels_true_adj, curr_labels_pred_adj)
        comp = completeness_score(curr_labels_true_adj, curr_labels_pred_adj)
        cluster_res1.append((perc_out, hom))
        cluster_res2.append((perc_out, comp))

        # ami = adjusted_mutual_info_score(curr_labels_true, curr_labels_pred)
        # num_in = float(len(curr_labels_pred))
        # perc_out = 1 - (num_in / tot_num)
        # print num_in, tot_num
        # print perc_out, ami
        cluster_res.append((perc_out, ami))
        # print "amiing", time.time() - start
        start = time.time()

    ## Saving file of tracks to cluster labels
    print("Saving pickled output", file=log.v5)
    start = time.time()
    output_file_name = self.output_folder + self.pre_saved_to_use + "extended_hom" + ".pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(cluster_res1, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)

    print("Saving pickled output", file=log.v5)
    start = time.time()
    output_file_name = self.output_folder + self.pre_saved_to_use + "extended_comp" + ".pkl"
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(cluster_res2, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)

    # ## Saving file of tracks to cluster labels
    # print >> log.v5, "Saving pickled output"
    # start = time.time()
    # output_file_name = self.output_folder + self.pre_saved_to_use + "one" + ".pkl"
    # with open(output_file_name, 'wb') as outputfile:
    #   cPickle.dump((ami,num_classes), outputfile)
    # print >> log.v5, "Saved, elapsed: ", time.time() - start

  def base_line_clustering(self,ys,tags,labels):
    from hdbscan import HDBSCAN, all_points_membership_vectors, membership_vector
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score
    ys = numpy.array(ys)
    orig_ys = ys.astype('float64')
    ys = orig_ys
    labels = numpy.array(labels)

    extra_res = dict()
    extra_res1 = dict()
    extra_res2 = dict()
    n_components = 128
    pca = PCA(n_components=n_components)
    ys = pca.fit_transform(orig_ys)

    from sklearn.cluster import KMeans
    num_clusters = 80

    # for num_clusters in (70,80,90,100):

    start = time.time()
    kmeans = KMeans(n_clusters=num_clusters, n_jobs=-1)
    cluster_labels = kmeans.fit_predict(ys)

    labels_pred = cluster_labels
    labels_true = labels

    ########################################################## Adjusted ################################
    labels_pred_adj = labels_pred[labels_true!=-1]
    labels_true_adj = labels_true[labels_true!=-1]

    ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
    hom = homogeneity_score(labels_true_adj, labels_pred_adj)
    comp = completeness_score(labels_true_adj, labels_pred_adj)
    extra_res1["kmeans"] = hom
    extra_res2["kmeans"] = comp
    print("kmeans:", ami, time.time() - start)
    extra_res["kmeans"] = ami

    from sklearn import mixture
    n_components = num_clusters
    start = time.time()
    # gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='tied')
    # gmm.fit(ys)
    # labels_pred= gmm.predict(ys)
    #
    # ########################################################## Adjusted ################################
    # labels_pred_adj = labels_pred[labels_true!=-1]
    # labels_true_adj = labels_true[labels_true!=-1]
    #
    # ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
    # print "gmm-tied:", ami, time.time() - start
    # extra_res["gmm-tied"] = ami

    # gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='diag')
    # gmm.fit(ys)
    # labels_pred = gmm.predict(ys)
    #
    # ########################################################## Adjusted ################################
    # labels_pred_adj = labels_pred[labels_true!=-1]
    # labels_true_adj = labels_true[labels_true!=-1]
    #
    # ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
    # print "gmm-diag:", ami, time.time() - start
    # extra_res["gmm-diag"] = ami

    # gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='spherical')
    # gmm.fit(ys)
    # labels_pred = gmm.predict(ys)
    #
    # ########################################################## Adjusted ################################
    # labels_pred_adj = labels_pred[labels_true!=-1]
    # labels_true_adj = labels_true[labels_true!=-1]
    #
    # ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
    # print "gmm-spherical:", ami, time.time() - start
    # extra_res["gmm-spherical"] = ami

    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(ys)
    labels_pred = gmm.predict(ys)

    ########################################################## Adjusted ################################
    labels_pred_adj = labels_pred[labels_true!=-1]
    labels_true_adj = labels_true[labels_true!=-1]

    ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
    hom = homogeneity_score(labels_true_adj, labels_pred_adj)
    comp = completeness_score(labels_true_adj, labels_pred_adj)
    extra_res1["gmm-full"] = hom
    extra_res2["gmm-full"] = comp
    print("gmm-full:", ami, time.time() - start)
    extra_res["gmm-full"] = ami

    ## Saving file of tracks to cluster labels
    print("Saving pickled output", file=log.v5)
    start = time.time()
    output_file_name = self.output_folder + "extra_plot_res_RCNN_hom_new.pkl"
    print(output_file_name)
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(extra_res1, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)

    ## Saving file of tracks to cluster labels
    print("Saving pickled output", file=log.v5)
    start = time.time()
    output_file_name = self.output_folder + "extra_plot_res_RCNN_comp_new.pkl"
    print(output_file_name)
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(extra_res2, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)

  def run_results_experiments(self,ys,tags,labels):
    from hdbscan import HDBSCAN,all_points_membership_vectors, membership_vector
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score
    ys = numpy.array(ys)
    orig_ys = ys.astype('float64')
    ys = orig_ys
    labels = numpy.array(labels)

    ## Optimise params:
    # num_outputs_already_saved = len(os.listdir(self.output_folder + "tree_cache/"))
    savedMemory = Memory(self.output_folder + "tree_cache/" + "0000" + "/")
    # savedMemory = Memory(cachedir=None, verbose=0)

    best1 = 0.0
    best1_arg = (0, 0, 0)
    best2 = 0.0
    best2_arg = (0, 0, 0)

    print(ys.shape)
    # n_components,min_samples,min_cluster_size = (128,14,14)
    n_components, min_samples, min_cluster_size = (128, 3, 3)
    pca = PCA(n_components=n_components)
    ys = pca.fit_transform(orig_ys)

    # save_file = self.output_folder + "save_info_new.txt"
    # fn = open(save_file, 'a')
    # fn.write("\n\n\n################NEW RUN###############\n\n\n")
    # fn.close()
    # for n_components in (128,):
    #
    #   pca = PCA(n_components=n_components)
    #   ys = pca.fit_transform(orig_ys)
    #
    #   for min_samples in (6,8,14,15,26,27,28,29,30,31,32,33,34,35):
    #     for min_cluster_size in (min_samples,):

    start = time.time()
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-2,
                    algorithm='boruvka_kdtree',
                    cluster_selection_method='eom', prediction_data=True, memory=savedMemory)

    print(ys.shape)

    cluster_labels = clusterer.fit_predict(ys)
    num_clusters = len(numpy.unique(cluster_labels))-1
    print("clustered", time.time()-start)
    cluster_idx = numpy.arange(len(cluster_labels))
    outlier_idx = cluster_idx[cluster_labels==-1]
    start = time.time()
    outlier_scores = clusterer.outlier_scores_
    outlier_scores = outlier_scores[outlier_idx]
    print("outliers", time.time() - start)
    start = time.time()
    # soft_clusters = all_points_membership_vectors(clusterer)
    # soft_clusters = soft_clusters[outlier_idx]
    soft_clusters = membership_vector(clusterer, ys[outlier_idx])
    weak_clusters = numpy.argmax(soft_clusters,-1)
    sort_idx = numpy.argsort(outlier_scores)
    outlier_idx = outlier_idx[sort_idx]
    weak_clusters = weak_clusters[sort_idx]
    num_points = len(cluster_labels)
    num_points_adj = len(labels[labels!=-1])
    print("weak clustering", time.time() - start)
    start = time.time()

    ### PLOTTING ##################

    labels_pred = cluster_labels[cluster_labels != -1]
    labels_true = labels[cluster_labels!=-1]
    # num_in = len(labels_pred)
    # perc_out = 1 - (float(num_in) / float(num_points))

    ########################################################## Adjusted ################################
    labels_pred_adj = labels_pred[labels_true!=-1]
    labels_true_adj = labels_true[labels_true!=-1]

    print(labels_pred_adj)
    print(labels_true[0:9])
    print(labels_true[10:19])
    print(labels_true[20:29])
    print(labels_true[30:39])

    num_in = len(labels_pred_adj)
    perc_out = 1 - (float(num_in) / float(num_points_adj))
    ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
    hom = homogeneity_score(labels_true_adj, labels_pred_adj)
    comp = completeness_score(labels_true_adj, labels_pred_adj)
    print(num_clusters)
    results = []
    results.append((perc_out,ami))
    results1 = []
    results1.append((perc_out, hom))
    results2 = []
    results2.append((perc_out, comp))

    # hom = homogeneity_score(labels_true_adj, labels_pred_adj)
    # comp = completeness_score(labels_true_adj, labels_pred_adj)
    # results1.append((perc_out, hom))
    # results2.append((perc_out, comp))

    print("preall", time.time() - start)
    start = time.time()

    for cc,(id,weak_cluster) in enumerate(zip(outlier_idx,weak_clusters)):

      cluster_labels[id] = weak_cluster
      if cc%80 == 79:
        labels_pred = cluster_labels[cluster_labels != -1]
        labels_true = labels[cluster_labels != -1]
        # num_in = len(labels_pred)
        # perc_out = 1 - (float(num_in) / float(num_points))

        ########################################################## Adjusted ################################
        labels_pred_adj = labels_pred[labels_true != -1]
        labels_true_adj = labels_true[labels_true != -1]

        num_in = len(labels_pred_adj)
        perc_out = 1 - (float(num_in) / float(num_points_adj))
        ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
        results.append((perc_out, ami))
        hom = homogeneity_score(labels_true_adj, labels_pred_adj)
        comp = completeness_score(labels_true_adj, labels_pred_adj)
        results1.append((perc_out, hom))
        results2.append((perc_out, comp))
        print(num_in, time.time() - start)

    labels_pred = cluster_labels[cluster_labels != -1]
    labels_true = labels[cluster_labels != -1]
    # num_in = len(labels_pred)
    # perc_out = 1 - (float(num_in) / float(num_points))

    ########################################################## Adjusted ################################
    labels_pred_adj = labels_pred[labels_true != -1]
    labels_true_adj = labels_true[labels_true != -1]

    num_in = len(labels_pred_adj)
    perc_out = 1 - (float(num_in) / float(num_points_adj))
    ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
    results.append((perc_out, ami))
    hom = homogeneity_score(labels_true_adj, labels_pred_adj)
    comp = completeness_score(labels_true_adj, labels_pred_adj)
    results1.append((perc_out, hom))
    results2.append((perc_out, comp))
    print(num_in, time.time() - start)

    print("all", time.time() - start)

    ## Saving file of tracks to cluster labels
    print("Saving pickled output", file=log.v5)
    start = time.time()
    output_file_name = self.output_folder + "plot_results_RCNN_hom_new.pkl"
    print(output_file_name)
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results1, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)

    ## Saving file of tracks to cluster labels
    print("Saving pickled output", file=log.v5)
    start = time.time()
    output_file_name = self.output_folder + "plot_results_RCNN_comp_new.pkl"
    print(output_file_name)
    with open(output_file_name, 'wb') as outputfile:
      pickle.dump(results2, outputfile)
    print("Saved, elapsed: ", time.time() - start, file=log.v5)

    start = time.time()

    result_to_show = numpy.array(results)
    import matplotlib.pyplot as plt
    plt.plot(result_to_show[:, 0], result_to_show[:, 1], '.')
    plt.show()

    result_to_show = numpy.array(results1)
    import matplotlib.pyplot as plt
    plt.plot(result_to_show[:,0],result_to_show[:,1],'.')
    plt.show()

    result_to_show = numpy.array(results2)
    import matplotlib.pyplot as plt
    plt.plot(result_to_show[:, 0], result_to_show[:, 1], '.')
    plt.show()

    print("plotting", time.time() - start)

          # fn = open(save_file, 'a')
          #
          # # start = time.time()
          # labels_pred = cluster_labels[cluster_labels!=-1]
          # labels_true = labels[cluster_labels!=-1]
          # # num_in = len(labels_pred)
          # # perc_out = 1 - (float(num_in) / float(num_points))
          #
          # ########################################################## Adjusted ################################
          # labels_pred_adj = labels_pred[labels_true!=-1]
          # labels_true_adj = labels_true[labels_true!=-1]
          #
          # num_in = len(labels_pred_adj)
          # perc_out = 1 - (float(num_in) / float(num_points_adj))
          # ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
          #
          # if ami>=best1:
          #   best1 = ami
          #   best1_arg = (n_components,min_samples, min_cluster_size)
          # infos = "\t".join(("ami1", str(perc_out), str(num_clusters), str((n_components,min_samples,min_cluster_size)), str(ami),str(best1_arg), str(best1), str(time.time() - start)))
          # print infos
          # fn.write(infos+"\n")
          #
          # cluster_labels[outlier_idx] = weak_clusters
          # labels_pred = cluster_labels[cluster_labels != -1]
          # labels_true = labels[cluster_labels != -1]
          # # num_in = len(labels_pred)
          # # perc_out = 1 - (float(num_in) / float(num_points))
          #
          # ########################################################## Adjusted ################################
          # labels_pred_adj = labels_pred[labels_true!=-1]
          # labels_true_adj = labels_true[labels_true!=-1]
          #
          # num_in = len(labels_pred_adj)
          # perc_out = 1 - (float(num_in) / float(num_points_adj))
          # ami = adjusted_mutual_info_score(labels_true_adj, labels_pred_adj)
          #
          # if ami>=best2:
          #   best2 = ami
          #   best2_arg = (n_components, min_samples, min_cluster_size)
          # infos =  "\t".join(("ami2", str(perc_out), "\t\t", str(num_clusters), str((n_components, min_samples,min_cluster_size)), str(ami),str(best2_arg), str(best2), str(time.time() - start)))
          # print infos
          # fn.write(infos+"\n")
          # print "\n"
          # fn.write("\n")
          #
          # fn.close()

    # quit()

  def forward(self, network, data, save_results=True, save_logits=False):
    self.network = network
    self.data = data

    # self.get_all_latents_from_faster_rcnn()
    # ys, tags, crop_ids, labels = self.get_RCNN_latents_from_file()
    # ys = numpy.array(ys)
    #
    # hyps = crop_ids["hyps"]
    # fn = crop_ids["tracklet_filenames"]
    # print str(hyps[0]), str(hyps[10]), str(hyps[100]), str(hyps[1000])
    # new_tags = [f+"___" + str(a) + "___" + "good_bye___again" for a,tag,f in zip(hyps,tags,fn)]
    # tags = new_tags
    # print tags[0].split("___")[:-2], tags[10].split("___")[:-2]

    # ys = tags = crop_ids = labels = None

    self.show_plots()
    quit()

    # if not self.use_pre_saved_data and not self.use_pre_extracted:
    #   ys, tags, crop_ids,labels = self.get_all_latents_from_network()
    # elif not self.use_pre_extracted:
    #   ys, tags, crop_ids,labels = self.get_all_latents_from_file()
    # else:
    #   ys = tags = crop_ids = labels = None

    # self.cluster_net_extension(ys,tags,labels)

    # quit()

    # self.run_results_experiments(ys,tags,labels)
    # self.base_line_clustering(ys, tags, labels)
    # self.cluster_net_extension(ys,tags,labels)

    # print labels[0:9]
    # print labels[10:19]
    # print labels[20:29]
    # print labels[30:39]

    # if not self.use_pre_extracted:
    #   original_track_ys, track_ims, track_classes, track_crop_ids, track_tags_name,max_classes, center_ids,track_labels = self.extract_centroids(ys,tags,labels)
    # else:
    #   original_track_ys, track_ims, track_classes, track_crop_ids, track_tags_name,max_classes, center_ids,track_labels = self.load_pre_extracted_centroids()

    # self.cluster_net_extension(original_track_ys, track_tags_name, track_labels)
    # quit()

    # print track_labels[0:9]
    # print track_labels[10:19]
    # print track_labels[20:29]
    # print track_labels[30:39]

    # quit()

    # track_labels = numpy.array(track_labels)
    # track_labels[track_labels==100] = -1
    # self.run_results_experiments(original_track_ys, track_classes, track_labels)
    # # self.base_line_clustering(original_track_ys,track_classes,track_labels)


    # unique_classes,track_labels = numpy.unique(track_classes, return_inverse=True)
    # unique_labels = numpy.unique(track_labels)
    # unknown_label = unique_labels[unique_classes=='unknown']
    # error_label = unique_labels[unique_classes=='tracking_error']
    # #
    # # num_unknowns = len(track_labels[track_labels==unknown_label])
    # # num_errors = len(track_labels[track_labels == error_label])
    # # num_tracks = len(track_labels)
    # # num_goods = num_tracks - num_unknowns - num_errors
    # #
    # # print num_tracks, num_goods, num_unknowns, num_errors
    # #
    # track_labels[track_labels==unknown_label] = -1
    # track_labels[track_labels == error_label] = -1
    # self.cluster_net_extension(original_track_ys, track_tags_name, track_labels)
    # # self.run_results_experiments(original_track_ys, track_classes, track_labels)
    # # self.base_line_clustering(original_track_ys, track_classes, track_labels)

    # quit()

    # if self.do_combine:
    #   self.combine_datasets()
    #
    # if self.do_cluster_all_options:
    #   self.cluster_all_options(original_track_ys)
    #
    # if self.do_optimize_params:
    #   self.optimize_params()
    #
    # self.run_one_whole_clustering(original_track_ys, track_classes, track_ims, track_tags_name, ys,tags)

    # self.run_all_forwarding_experiements(ys, tags, original_track_ys, track_tags_name, track_classes,max_classes)

    # # curr_file = "kitti_train-merged_16_8_16"
    # # curr_test = "2c-ClusMax"
    # # curr_type = "merged"
    # curr_file = "kitti_train-post_inference-16_8_16"
    # curr_test = "1a-DetInd"
    # curr_type = "post_inference"
    # command = "/home/luiten/vision/savitar/scripts/tracklets/remap_and_eval_tracking_KITTI_train.sh /home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/assigned_output/%s/%s.pkl 2017_11_06_1444_54_mining %s_RCNN_savitar | grep 'MODA\|False Positives\|Missed Targets' | awk '{print $NF}' | sed ':a;N;$!ba;s/\\n/\\t/g'"%(curr_file,curr_test,curr_type)
    # var = os.popen(command).read()
    # print "from python:", var

    # /home/luiten/vision/savitar/scripts/tracklets/remap_and_eval_tracking_KITTI_train.sh /home/luiten/vision/savitar/forwarded/COCO_Similarity_Triplet_Edit/assigned_output/kitti_train-post_inference-16_8_16/1a-DetInd.pkl 2017_11_06_1444_54_mining post_inference_RCNN_savitar | grep 'MODA\|False Positives\|Missed Targets' | awk '{print $NF}' | sed ':a;N;$!ba;s/\n/\t/g'



    # ## Combining clustering and detection labels
    # detection_labels, detection_cluster_ids = numpy.unique(track_classes, return_inverse=True)
    # curr_cluster_ids = idx_array[cluster_ids == -1]
    # curr_det_labels = [track_classes[a] for a in curr_cluster_ids]
    # for c,id in zip(curr_det_labels,curr_cluster_ids):
    #   if c != 'unknown':
    #     for ii,n in enumerate(names):
    #       if c==n:
    #         cluster_ids[id] = ii+1

