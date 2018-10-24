import os
import sys
import numpy
import tensorflow as tf

from ReID_net.Engine import Engine
from ReID_net.Config import Config
from ReID_net.main import init_log
from ReID_net.Log import log
from ReID_net.datasets.Util.Util import smart_shape


def compute_ious(x, y):
  y = y[:, numpy.newaxis]
  min_ = numpy.minimum(x, y)
  max_ = numpy.maximum(x, y)
  I = numpy.maximum(min_[..., 1] - max_[..., 0], 0) * numpy.maximum(min_[..., 3] - max_[..., 2], 0)
  area_x = (x[..., 1] - x[..., 0]) * (x[..., 3] - x[..., 2])
  area_y = (y[..., 1] - y[..., 0]) * (y[..., 3] - y[..., 2])
  U = area_x + area_y - I
  assert (U != 0).all()
  IOUs = I / U
  return IOUs


def create_extractor(config_path, extract_fn):
  assert os.path.exists(config_path), config_path
  config = Config(config_path)
  init_log(config)
  config.initialize()
  # dump the config into the log
  print(open(config_path).read(), file=log.v4)
  engine = Engine(config)
  network = engine.test_network
  output_layer = network.get_output_layer()
  return extract_fn(engine, output_layer)


def detection_extractor(engine, output_layer):
  outputs = output_layer.outputs[0]
  det_boxes, det_scores, reid, det_classes, num_detections = outputs
  data = engine.valid_data

  def extract(filename):
    feed_dict = {data.img_filename_placeholder: filename}
    return engine.session.run([det_boxes, det_scores, det_classes, num_detections], feed_dict=feed_dict)
  return extract


def clustering_features_extractor(engine, output_layer):
  outputs = output_layer.outputs[0]
  features = output_layer.y_class_features
  det_boxes, det_scores, reid, det_classes, num_detections = outputs
  det_boxes = tf.squeeze(det_boxes, axis=2)
  #conceptual problem: the features before the softmax are shared across anchors
  #let's just ignore that for now and replicate the features for each anchor
  n_anchors = 9
  shape = smart_shape(features)
  features = tf.tile(tf.expand_dims(features, axis=2), multiples=[1, 1, n_anchors, 1])
  features = tf.reshape(features, tf.stack([shape[0], -1, shape[-1]], axis=0))

  def extract(filename, extract_boxes):
    data = engine.valid_data
    feed_dict = {data.img_filename_placeholder: filename, data.bboxes_placeholder: extract_boxes}
    det_boxes_val, det_features_val, det_scores_val = engine.session.run([det_boxes, features, det_scores],
                                                                         feed_dict=feed_dict)
    #remove batch dimension
    batch_size = det_boxes_val.shape[0]
    assert batch_size == 1
    det_boxes_val = det_boxes_val[0]
    det_features_val = det_features_val[0]
    det_scores_val = det_scores_val[0]
    #compute IOUs and select most overlapping, for convenience let's do this with numpy instead of tensorflow
    ious = compute_ious(det_boxes_val, extract_boxes)
    indices = ious.argmax(axis=1)
    det_features_out = det_features_val[indices]
    det_scores_out = det_scores_val[indices]
    return det_features_out, det_scores_out
  return extract


def faster_rcnn_extractor(engine, output_layer):
  roi_placeholder = output_layer.roi_placeholder
  scores = output_layer.classification_outputs
  features = output_layer.classification_features
  data = engine.valid_data

  def extract(filename, extract_boxes):
    feed_dict = {data.img_filename_placeholder: filename, roi_placeholder: extract_boxes}
    scores_val, features_val = engine.session.run([scores, features], feed_dict=feed_dict)
    return scores_val, features_val
  return extract


def test_clustering_features_extractor():
  assert len(sys.argv) == 2
  test_img = "/work/voigtlaender/data/KITTI/training/image_2/005592.png"

  # detection_extractor = create_extractor(sys.argv[1], detection_extractor)
  # print detection_extractor(test_img)

  extractor = create_extractor(sys.argv[1], clustering_features_extractor)
  # input and output bboxes are encoded as [top, bottom, left, right]
  test_bboxes = numpy.array([[178, 200, 781, 838], [10, 100, 20, 150]], dtype="float32")
  feats, scores = extractor(test_img, test_bboxes)
  # feats are 256 dimensional features for each input boundin box.
  # note that the features used to compute the class scores are shared over anchors, which might be a problem
  # if you need class labels, you can argmax the scores
  print(feats.shape, scores.shape)


def test_faster_rcnn_extractor():
  assert len(sys.argv) == 2
  test_img = "/work/voigtlaender/data/KITTI/training/image_2/005592.png"
  extractor = create_extractor(sys.argv[1], faster_rcnn_extractor)
  # input and output bboxes are encoded as [top, bottom, left, right]
  test_bboxes = numpy.array([[178, 200, 781, 838], [10, 100, 20, 150]], dtype="float32")
  scores, feats = extractor(test_img, test_bboxes)
  print(scores.shape, feats.shape)

if __name__ == "__main__":
  #test_clustering_features_extractor()
  test_faster_rcnn_extractor()
