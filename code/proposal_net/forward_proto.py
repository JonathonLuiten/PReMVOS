import tensorflow as tf
import numpy as np
import cv2

from hypotheses_pb2 import HypothesisSet
import config
from eval import detect_one_image


def forward_protobuf(pred_func, output_folder, forward_dataset, generic_images_folder, generic_images_pattern):
    tf.gfile.MakeDirs(output_folder)
    hypo_set = HypothesisSet()
    with open("/work/merkelbach/datasets/dashcam_mining/tracking_results_raw/vid_01/eval/12-Mar-2018--19-25-00vayV/hypos_protobuf/hypotheses_500frames_1860hyp___work__pv182253__data__dashcam__dashcam_videos_frames__vid_01__frames_cropped__clip_005.hypset", "rb") as f:
        hypo_set.ParseFromString(f.read())
    for hyp in hypo_set.hypotheses:
        boxes = hyp.bounding_boxes_2D_with_timestamps
        for t, box in boxes.items():
            img_filename = "/work/merkelbach/datasets/dashcam_mining/videos/vid_01/frames_cropped/all_frames/video_0001_frames" + "%09d" % t + ".png"
            print(img_filename, box)
            img_val = cv2.imread(img_filename, cv2.IMREAD_COLOR)
            assert config.PROVIDE_BOXES_AS_INPUT
            input_boxes = np.array([[box.x0, box.y0, box.x0 + box.w, box.y0 + box.h]], dtype=np.float32)
            results = detect_one_image(img_val, pred_func, input_boxes)
            print(results)
