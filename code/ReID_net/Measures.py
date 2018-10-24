import numpy
import tensorflow as tf
from sklearn.metrics import confusion_matrix, average_precision_score

import ReID_net.Constants as Constants
from ReID_net.Log import log


def create_confusion_matrix(pred, targets, n_classes):
  result = None
  targets = targets.reshape((targets.shape[0], -1))
  pred = pred.reshape((pred.shape[0], -1))
  for i in range(pred.shape[0]):
    conf_matrix = confusion_matrix(targets[i],
                                   pred[i],
                                   list(range(0, n_classes)))
    conf_matrix = conf_matrix[numpy.newaxis, :, :]

    if result is None:
      result = conf_matrix
    else:
      result = numpy.append(result, conf_matrix, axis=0)
  return result


def get_average_precision(targets, outputs, conf_matrix):
  targets = targets.reshape(targets.shape[0], -1)
  outputs = outputs[:, :, :, :, 1]
  outputs = outputs.reshape(outputs.shape[1], -1)

  ap = numpy.empty(outputs.shape[0], numpy.float64)
  # ap_interpolated = numpy.empty(outputs.shape[0], numpy.float64)
  for i in range(outputs.shape[0]):
    # precision, recall, thresholds = precision_recall_curve(targets[i], outputs[i])
    ap[i] = average_precision_score(targets[i].flatten(), outputs[i].flatten())
    # result = eng.get_ap(matlab.double(outputs[i].tolist()), matlab.double(targets[i].tolist()))
    # ap_interpolated[i] = result

  ap = numpy.nan_to_num(ap)
  # ap_interpolated = numpy.nan_to_num(ap_interpolated)
  return ap


def compute_binary_ious_tf(targets, outputs):
  binary_ious = [compute_iou_for_binary_segmentation(target, output) for target, output in
                 zip(targets, outputs)]
  return numpy.sum(binary_ious, dtype="float32")


def compute_iou_for_binary_segmentation(y_argmax, target):
  I = numpy.logical_and(y_argmax == 1, target == 1).sum()
  U = numpy.logical_or(y_argmax == 1, target == 1).sum()
  if U == 0:
    IOU = 1.0
  else:
    IOU = float(I) / U
  return IOU


def compute_measures_for_binary_segmentation(prediction, target):
  T = target.sum()
  P = prediction.sum()
  I = numpy.logical_and(prediction == 1, target == 1).sum()
  U = numpy.logical_or(prediction == 1, target == 1).sum()

  if U == 0:
    recall = 1.0
    precision = 1.0
    iou = 1.0
  else:
    if T == 0:
      recall = 1.0
    else:
      recall = float(I) / T

    if P == 0:
      precision = 1.0
    else:
      precision = float(I) / P

    iou = float(I) / U

  measures = {"recall": recall, "precision": precision, "iou": iou}
  return measures


def average_measures(measures_dicts):
  keys = list(measures_dicts[0].keys())
  averaged_measures = {}
  for k in keys:
    vals = [m[k] for m in measures_dicts]
    val = numpy.mean(vals)
    averaged_measures[k] = val
  return averaged_measures


def compute_iou_from_logits(preds, labels, num_labels):
  """
  Computes the intersection over union (IoU) score for given logit tensor and target labels
  :param logits: 4D tensor of shape [batch_size, height, width, num_classes]
  :param labels: 3D tensor of shape [batch_size, height, width] and type int32 or int64
  :param num_labels: tensor with the number of labels
  :return: 1D tensor of shape [num_classes] with intersection over union for each class, averaged over batch
  """
  with tf.variable_scope("IoU"):
    # compute predictions
    # probs = softmax(logits, axis=-1)
    # preds = tf.arg_max(probs, dimension=3)
    # num_labels = preds.get_shape().as_list()[-1];
    IoUs = []
    for label in range(num_labels):
      # find pixels with given label
      P = tf.equal(preds, label)
      L = tf.equal(labels, label)
      # Union
      U = tf.logical_or(P, L)
      U = tf.reduce_sum(tf.cast(U, tf.float32))
      # intersection
      I = tf.logical_and(P, L)
      I = tf.reduce_sum(tf.cast(I, tf.float32))

      IOU = tf.cast(I, tf.float32) / tf.cast(U, tf.float32)
      # U might be 0!
      IOU = tf.where(tf.equal(U, 0), 1, IOU)
      IOU = tf.Print(IOU, [IOU], "iou" + repr(label))
      IoUs.append(IOU)
    return tf.reshape(tf.stack(IoUs), (num_labels,))


def calc_measures_avg(measures, n_imgs, ignore_classes, for_final_result):
  measures_result = {}
  # these measures can just be averaged
  for measure in [Constants.ERRORS, Constants.IOU, Constants.BINARY_IOU, Constants.AP, Constants.MOTA, Constants.MOTP,
                  Constants.AP_INTERPOLATED, Constants.FALSE_POSITIVES, Constants.FALSE_NEGATIVES,
                  Constants.ID_SWITCHES]:
    if measure in measures:
      measures_result[measure] = numpy.sum(measures[measure]) / n_imgs

  # TODO: This has to be added as IOU instead of conf matrix.
  if Constants.CONFUSION_MATRIX in measures:
    measures_result[Constants.IOU] = calc_iou(measures, n_imgs, ignore_classes)

  if Constants.CLICKS in measures:
    clicks = [int(x.rsplit(':', 1)[-1]) for x in measures[Constants.CLICKS]]
    measures_result[Constants.CLICKS] = float(numpy.sum(clicks)) / n_imgs

  if for_final_result and Constants.DETECTION_AP in measures:
    from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluation
    if isinstance(measures[Constants.DETECTION_AP], ObjectDetectionEvaluation):
      evaluator = measures[Constants.DETECTION_AP]
    else:
      n_classes = measures[Constants.DETECTION_AP][-2]
      evaluator = ObjectDetectionEvaluation(n_classes, matching_iou_threshold=0.5)
      evaluator.next_image_key = 0  # add a new field which we will use
      _add_aps(evaluator, measures[Constants.DETECTION_AP])

    aps, mAP, _, _, _, _ = evaluator.evaluate()
    measures_result[Constants.DETECTION_APS] = aps
    measures_result[Constants.DETECTION_AP] = mAP

  if for_final_result and Constants.CLUSTER_IDS in measures and Constants.ORIGINAL_LABELS in measures:
    from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score
    labels_true = numpy.reshape(numpy.array(measures[Constants.ORIGINAL_LABELS], dtype=numpy.int32), [-1])
    labels_pred = numpy.reshape(numpy.array(measures[Constants.CLUSTER_IDS], dtype=numpy.int32), [-1])
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    measures_result[Constants.ADJUSTED_MUTUAL_INFORMATION] = ami
    homogeneity = homogeneity_score(labels_true, labels_pred)
    measures_result[Constants.HOMOGENEITY] = homogeneity
    completeness = completeness_score(labels_true, labels_pred)
    measures_result[Constants.COMPLETENESS] = completeness

  NO_EVAL = False
  if  not NO_EVAL:
    if for_final_result and Constants.ORIGINAL_LABELS in measures and Constants.EMBEDDING in measures:
      from sklearn import mixture
      from sklearn.cluster import KMeans
      from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, completeness_score
      embeddings = numpy.array(measures[Constants.EMBEDDING], dtype=numpy.int32)
      embeddings = numpy.reshape(embeddings,[-1,embeddings.shape[-1]])
      labels_true = numpy.reshape(numpy.array(measures[Constants.ORIGINAL_LABELS], dtype=numpy.int32), [-1])
      # n_components = 80
      # n_components = 400
      # n_components = 1000
      n_components = 3000
      import time

      # start = time.time()
      # gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
      # gmm.fit(embeddings)
      # labels_pred= gmm.predict(embeddings)
      # print "gmm took ", time.time()-start

      start = time.time()
      kmeans = KMeans(n_clusters=n_components, n_jobs=-1)
      labels_pred = kmeans.fit_predict(embeddings)
      print("km took ", time.time() - start)

      ami = adjusted_mutual_info_score(labels_true, labels_pred)
      measures_result[Constants.ADJUSTED_MUTUAL_INFORMATION] = ami
      homogeneity = homogeneity_score(labels_true, labels_pred)
      measures_result[Constants.HOMOGENEITY] = homogeneity
      completeness = completeness_score(labels_true, labels_pred)
      measures_result[Constants.COMPLETENESS] = completeness

  return measures_result


def calc_iou(measures, n_imgs, ignore_classes):
  assert Constants.CONFUSION_MATRIX in measures
  conf_matrix = measures[Constants.CONFUSION_MATRIX]
  assert conf_matrix.shape[0] == n_imgs  # not sure, if/why we need these n_imgs

  ious = get_ious_per_image(measures, ignore_classes)
  IOU_avg = numpy.mean(ious)
  return IOU_avg


def get_ious_per_image(measures, ignore_classes):
  assert Constants.CONFUSION_MATRIX in measures
  conf_matrix = measures[Constants.CONFUSION_MATRIX]

  I = (numpy.diagonal(conf_matrix, axis1=1, axis2=2)).astype("float32")
  sum_predictions = numpy.sum(conf_matrix, axis=1)
  sum_labels = numpy.sum(conf_matrix, axis=2)
  U = sum_predictions + sum_labels - I
  n_classes = conf_matrix.shape[-1]
  class_mask = numpy.ones((n_classes,))
  # Temporary fix to avoid index out of bounds when there is a void label in the list of classes to be ignored.
  ignore_classes = numpy.array(ignore_classes)
  ignore_classes = ignore_classes[numpy.where(ignore_classes <= n_classes)]
  class_mask[ignore_classes] = 0

  ious = []
  for i, u in zip(I, U):
    mask = numpy.logical_and(class_mask, u != 0)
    if mask.any():
      iou = (i[mask] / u[mask]).mean()
    else:
      print("warning, mask only consists of ignore_classes", file=log.v5)
      iou = 1.0
    ious.append(iou)

  return ious


def _add_aps(evaluator, aps):
  det_boxes, det_scores, det_classes, num_detections, gt_boxes, gt_classes, gt_ids, n_classes, tags = aps
  for det_boxes_im, det_scores_im, det_classes_im, num_detections_im, gt_boxes_im, \
          gt_classes_im, gt_ids_im in zip(det_boxes, det_scores, det_classes, num_detections, gt_boxes, gt_classes,
                                          gt_ids):
    # TODO: get the actual image names or hashes instead of a running number?
    # TODO: add is_difficult for gt
    image_key = evaluator.next_image_key
    evaluator.add_single_ground_truth_image_info(image_key,
                                                 (gt_boxes_im[gt_ids_im > 0][:, [0, 2, 1, 3]]).astype("float32"),
                                                 gt_classes_im[gt_ids_im > 0])
    evaluator.add_single_detected_image_info(image_key, det_boxes_im[:num_detections_im, [0, 2, 1, 3]],
                                             det_scores_im[:num_detections_im], det_classes_im[:num_detections_im])
    evaluator.next_image_key += 1


def calc_measures_sum(measures1, measures2):
  measures_result = {}

  if not measures1:
    return measures2

  if not measures2:
    return measures1

  # these measures can just be added
  for measure in [Constants.ERRORS, Constants.IOU, Constants.BINARY_IOU, Constants.AP, Constants.MOTA, Constants.MOTP,
                  Constants.AP_INTERPOLATED, Constants.FALSE_POSITIVES, Constants.FALSE_NEGATIVES,
                  Constants.ID_SWITCHES, Constants.ORIGINAL_LABELS, Constants.CLUSTER_IDS, Constants.EMBEDDING]:
    if measure in measures1 and measure in measures2:
      measures_result[measure] = measures1[measure] + measures2[measure]

  if Constants.CONFUSION_MATRIX in measures1 and Constants.CONFUSION_MATRIX in measures2:
    conf_matrix1 = measures1[Constants.CONFUSION_MATRIX]
    conf_matrix2 = measures2[Constants.CONFUSION_MATRIX]
    if isinstance(conf_matrix1, tf.Tensor) and isinstance(conf_matrix2, tf.Tensor):
      measures_result[Constants.CONFUSION_MATRIX] = tf.concat([conf_matrix2, conf_matrix1], axis=0)
    else:
      measures_result[Constants.CONFUSION_MATRIX] = numpy.append(conf_matrix2, conf_matrix1, axis=0)

  if Constants.CLICKS in measures1 and Constants.CLICKS in measures2:
    measures_result[Constants.CLICKS] = numpy.append(measures1[Constants.CLICKS], measures2[Constants.CLICKS])

  if Constants.DETECTION_AP in measures1 and Constants.DETECTION_AP in measures2:
    from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluation
    ap1 = measures1[Constants.DETECTION_AP]
    ap2 = measures2[Constants.DETECTION_AP]
    assert not (isinstance(ap1, ObjectDetectionEvaluation) and isinstance(ap2, ObjectDetectionEvaluation))
    if isinstance(ap1, ObjectDetectionEvaluation):
      evaluator = ap1
      new_aps = [ap2]
    elif isinstance(ap2, ObjectDetectionEvaluation):
      evaluator = ap2
      new_aps = [ap1]
    else:
      n_classes = ap1[-2]
      evaluator = ObjectDetectionEvaluation(n_classes, matching_iou_threshold=0.5)
      evaluator.next_image_key = 0  # add a new field which we will use
      new_aps = [ap1, ap2]
    for aps in new_aps:
      _add_aps(evaluator, aps)
    measures_result[Constants.DETECTION_AP] = evaluator
  return measures_result


def get_error_string(measures, task):
  result_string = ""

  if task == "train":
    result_string += "train_err:"
  else:
    result_string += "valid_err:"

  # handle a few special cases
  if Constants.CONFUSION_MATRIX in measures:
    result_string += "(IOU) %4f" % measures[Constants.CONFUSION_MATRIX]

  if Constants.ERRORS in measures:
    result_string += "  %4f" % measures[Constants.ERRORS]

  if Constants.RANKS in measures:
    result_string += " " + measures[Constants.RANKS]

  if Constants.CLICKS in measures:
    result_string += " (Avg Clicks): %4f" % measures[Constants.CLICKS]

  if Constants.DETECTION_APS in measures:
    result_string += "(aps): " + str(measures[Constants.DETECTION_APS])

  # then handle other cases uniformly
  for measure in [Constants.IOU, Constants.BINARY_IOU, Constants.AP, Constants.MOTA, Constants.MOTP,
                  Constants.AP_INTERPOLATED, Constants.FALSE_POSITIVES, Constants.FALSE_NEGATIVES,
                  Constants.ID_SWITCHES, Constants.DETECTION_AP,
                  Constants.ADJUSTED_MUTUAL_INFORMATION, Constants.HOMOGENEITY, Constants.COMPLETENESS]:
    if measure in measures:
      result_string += " (%s) %4f" % (measure, measures[measure])

  return result_string


def calc_ap_from_cm(conf_matrix):
  tp = conf_matrix[1][1]
  fp = conf_matrix[0][1]
  fn = conf_matrix[1][0]

  precision = tp.astype(float) / (tp + fp).astype(float)
  recall = tp.astype(float) / (tp + fn).astype(float)

  return precision, recall
