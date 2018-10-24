from scipy.misc import imread
from pycocotools.mask import encode, iou, area, decode, toBbox
import numpy as np

from refinement_net.core.Engine import Engine
from refinement_net.core.Config import Config
from refinement_net.core.Log import log
from refinement_net.core import Extractions
import refinement_net.datasets.DataKeys as DataKeys


def init_log(config):
  log_dir = "../output/logs/refinement_net"
  model = config.string("model")
  filename = log_dir + model + ".log"
  verbosity = config.int("log_verbosity", 3)
  log.initialize([filename], [verbosity], [])

def refinement_net_init():
  config_path = "refinement_net/configs/live"
  config = Config(config_path)
  init_log(config)
  engine = Engine(config)
  return engine

def extract(key,extractions):
  if key not in extractions:
    return None
  val = extractions[key]
  # for now assume we only use 1 gpu for forwarding
  assert len(val) == 1, len(val)
  val = val[0]
  # # for now assume, we use a batch size of 1 for forwarding
  assert val.shape[0] == 1, val.shape[0]
  val = val[0]
  return val

def do_refinement(proposals,image_fn,refinement_net):
  image = imread(image_fn)
  boxes = [prop['bbox'] for prop in proposals]
  data = refinement_net.valid_data
  image_data = data.set_up_data_for_image(image, boxes)

  for idx in range(len(boxes)):
    feed_dict = data.get_feed_dict_for_next_step(image_data, idx)
    step_res = refinement_net.trainer.validation_step(feed_dict=feed_dict, extraction_keys=[Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE,
                                                                                            Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE, DataKeys.OBJ_TAGS])
    extractions = step_res[Extractions.EXTRACTIONS]
    predicted_segmentation = extract(Extractions.SEGMENTATION_MASK_ORIGINAL_SIZE,extractions)
    obj_tag = extract(DataKeys.OBJ_TAGS,extractions)
    obj_tag = int(obj_tag.decode('utf-8'))
    mask = predicted_segmentation.astype("uint8") * 255
    encoded_mask = encode(np.asfortranarray(mask))
    encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")
    proposals[obj_tag]["segmentation"] = encoded_mask

    posteriors = extract(Extractions.SEGMENTATION_POSTERIORS_ORIGINAL_SIZE,extractions)
    conf_scores = posteriors.copy()
    conf_scores[predicted_segmentation == 0] = 1 - posteriors[predicted_segmentation == 0]
    conf_scores = 2 * conf_scores - 1
    conf_score = conf_scores[:].mean()
    proposals[obj_tag]["conf_score"] = str(conf_score)


  return(proposals)