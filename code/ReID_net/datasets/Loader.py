import ReID_net.Constants as Constants
from .COCO.COCO import COCODataset
from .DAVIS.DAVIS import DAVISDataset, DAVIS2017Dataset
from ReID_net.datasets.COCO.COCO_detection import COCODetectionDataset
from ReID_net.datasets.COCO.COCO_instance import COCOInstanceDataset
from ReID_net.datasets.COCO.COCO_interactive import COCOInteractiveDataset
from ReID_net.datasets.COCO.COCO_objectness import CocoObjectnessDataset
from ReID_net.datasets.Custom.Custom import CustomDataset
from ReID_net.datasets.DAVIS.DAVIS2017_oneshot import Davis2017OneshotDataset
from ReID_net.datasets.DAVIS.DAVIS_instance import DAVISInstanceDataset
from ReID_net.datasets.DAVIS.DAVIS_iterative import DavisIterativeDataset
from ReID_net.datasets.DAVIS.DAVIS_masktransfer import DAVISMaskTransferDataset
from ReID_net.datasets.DAVIS.DAVIS_oneshot import DavisOneshotDataset
from ReID_net.datasets.Similarity.COCO_Similarity import COCOSimilarityDataset
from ReID_net.datasets.Similarity.DAVIS_Similarity import DAVISSimilarityDataset
from ReID_net.datasets.Similarity.DAVIS_Forward_Similarity import DAVISForwardSimilarityDataset
from ReID_net.datasets.Similarity.DAVIS_Forward_Feed import DAVISForwardFeedDataset

def load_dataset(config, subset, session, coordinator, name=None):
  if name is None:
    if subset != "train":
      name = config.str("dataset_val", "").lower()
    if name is None or name == "":
      name = config.str("dataset").lower()
  task = config.str("task", "")
  if task in ("oneshot", "oneshot_forward", "online", "offline"):
    if name == "davis":
      return DavisOneshotDataset(config, subset, use_old_label=False)
    elif name == "davis_instance":
      return DAVISInstanceDataset(config, subset, coordinator)
    elif name == "davis_mask":
      return DavisOneshotDataset(config, subset, use_old_label=True)
    elif name in ("davis17", "davis2017"):
      return Davis2017OneshotDataset(config, subset)
    else:
      assert False, "Unknown dataset for oneshot: " + name

  if task == "forward" and name == "davis_mask":
    return DavisOneshotDataset(config, subset, use_old_label=True)
  if task == Constants.ONESHOT_INTERACTIVE:
    return InteractiveOneShot(config, subset, video_data_dir=None)
  if task == Constants.ITERATIVE_FORWARD and name == "davis":
    return DavisIterativeDataset(config, subset, use_old_label=True)

  if name == "davis":
    return DAVISDataset(config, subset, coordinator)
  elif name == "davis_instance":
    return DAVISInstanceDataset(config, subset, coordinator)
  elif name in ("davis17", "davis2017"):
    return DAVIS2017Dataset(config, subset, coordinator)
  elif name in ("davis17_test", "davis2017_test"):
    return DAVIS2017Dataset(config, subset, coordinator, fraction=0.002)
  elif name == "davis_test":
    return DAVISDataset(config, subset, coordinator, fraction=0.05)
  elif name == "davis_mask":
    return DAVISMaskTransferDataset(config, subset, coordinator)
  elif name == "davis_mask_test":
    return DAVISMaskTransferDataset(config, subset, coordinator, fraction=0.02)
  elif name == "coco":
    return COCODataset(config, subset, coordinator)
  elif name == "coco_test":
    return COCODataset(config, subset, coordinator, fraction=0.001)
  elif name == "coco_objectness":
    return CocoObjectnessDataset(config, subset, coordinator)
  elif name == "coco_objectness_test":
    return CocoObjectnessDataset(config, subset, coordinator, fraction=0.001)
  elif name == "coco_instance":
    return COCOInstanceDataset(config, subset, coordinator)
  elif name == "coco_instance_test":
    return COCOInstanceDataset(config, subset, coordinator, fraction=0.001)
  elif name == "coco_interactive":
    if subset == "valid":
      dataset = COCOInteractiveDataset(config, subset, coordinator, fraction=0.2)
    else:
      dataset = COCOInteractiveDataset(config, subset, coordinator)
    return dataset
  elif name == "coco_interactive_test":
    return COCOInteractiveDataset(config, subset, coordinator, fraction=0.001)
  elif name == "coco_detection":
    return COCODetectionDataset(config, subset, coordinator)
  elif name == "coco_detection_test":
    return COCODetectionDataset(config, subset, coordinator, fraction=0.01)
  elif name == "coco_similarity":
    return COCOSimilarityDataset(config, subset, coordinator)
  elif name == "davis_similarity":
    return DAVISSimilarityDataset(config, subset, coordinator)
  elif name == "davis_forward_similarity":
    return DAVISForwardSimilarityDataset(config, subset, coordinator)
  elif name == "davis_forward_feed":
    return DAVISForwardFeedDataset(config, subset, coordinator)
  elif name == "custom":
    return CustomDataset(config, subset, coordinator)
  elif name == "none":
    return None
  assert False, "Unknown dataset " + name
