import ReID_net.Constants as Constants
from ReID_net.Forwarding.ClusteringForwarder import ClusteringForwarder
from ReID_net.Forwarding.ReIDForwarding import ReIDForwarder
from ReID_net.Forwarding.Forwarder import ImageForwarder
from ReID_net.Forwarding.InteractiveImageForwarder import InteractiveImageForwarder
from ReID_net.Forwarding.MOT15Forwarder import MOT15Forwarder
from ReID_net.Forwarding.MARKET1501Forwarder import MARKET1501Forwarder
from ReID_net.Forwarding.MaskTransferForwarder import MaskTransferForwarder, MaskTransferOneshotForwarder
from ReID_net.Forwarding.OfflineAdaptingForwarder import OfflineAdaptingForwarder
from ReID_net.Forwarding.OneshotForwarder import OneshotForwarder
from ReID_net.Forwarding.OnlineAdaptingForwarder import OnlineAdaptingForwarder
from ReID_net.Forwarding.PascalVOCInstanceForwarder import PascalVOCInstanceForwarder
from ReID_net.Forwarding.COCOInstanceForwarder import COCOInstanceForwarder
from ReID_net.datasets.InteractiveEval import InteractiveEval


def forward(engine, network, data, dataset_name, save_results, save_logits):
  if dataset_name == "davis_mask":
    forwarder = MaskTransferForwarder(engine)
  elif dataset_name in ("mot", "mot15"):
    forwarder = MOT15Forwarder(engine)
  elif dataset_name in ("market", "market1501"):
    forwarder = MARKET1501Forwarder(engine)
  elif (dataset_name.startswith("pascalvoc_instance", 0, len(dataset_name)) or
          dataset_name.startswith("pascalvoc_interactive", 0, len(dataset_name))):
    forwarder = PascalVOCInstanceForwarder(engine)
  elif dataset_name == "coco_instance":
    forwarder = COCOInstanceForwarder(engine)
  else:
    forwarder = ImageForwarder(engine)
  forwarder.forward(network, data, save_results, save_logits)


def forward_clustering(engine, network, data):
  forwarder = ClusteringForwarder(engine)
  forwarder.forward(network, data)

def forward_reid(engine,network,data):
  forwarder = ReIDForwarder(engine)
  forwarder.forward(network, data)


def oneshot_forward(engine, save_results, save_logits):
  if engine.dataset in ("davis", "davis_instance", "davis17", "davis2017", "davis_video", "oxford", "youtube", "youtubeobjects",
                        "youtubefull", "youtubeobjectsfull", "segtrackv2"):
    forwarder = OneshotForwarder(engine)
  elif engine.dataset == "davis_mask":
    forwarder = MaskTransferOneshotForwarder(engine)
  else:
    assert False, "unknown dataset for oneshot: " + engine.dataset
  forwarder.forward(None, None, save_results, save_logits)


def online_forward(engine, save_results, save_logits):
  forwarder = OnlineAdaptingForwarder(engine)
  forwarder.forward(None, None, save_results, save_logits)


def offline_forward(engine, save_results, save_logits):
  forwarder = OfflineAdaptingForwarder(engine)
  forwarder.forward(None, None, save_results, save_logits)


def interactive_forward(engine, network, data, save_results, save_logits, task):

  def mask_generation_fn(engine_to_use, img, tag, label, u0, u1, old_label=None):
    forwarder = InteractiveImageForwarder(engine_to_use)
    data_to_use = engine_to_use.valid_data
    data_to_use.create_feed_dict(img=img, tag=tag, label=label,
                          u0=u0, u1=u1, old_label=old_label)
    ys_argmax_values = forwarder.forward(network=engine_to_use.test_network, data=data_to_use,
                                         save_results=save_results, save_logits=save_logits)

    return ys_argmax_values


  eval_ = InteractiveEval(engine, mask_generation_fn)
  eval_.eval()
