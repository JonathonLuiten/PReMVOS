from scipy.misc import imread
import matplotlib.pyplot as plt

from ReID_net.Engine import Engine
from ReID_net.Config import Config
from ReID_net.Log import log

def init_log(config):
  log_dir = "../output/logs/ReID_net"
  model = config.str("model")

  print(log_dir,model,".log")

  filename = log_dir + model + ".log"
  verbosity = config.int("log_verbosity", 3)
  log.initialize([filename], [verbosity], [])

def ReID_net_init():
  config_path = "ReID_net/configs/live"
  config = Config(config_path)
  init_log(config)
  config.initialize()
  engine = Engine(config)
  return engine

def add_ReID(proposals,image_fn,ReID_net):
  image = imread(image_fn)
  boxes = [prop['bbox'] for prop in proposals]
  network = ReID_net.test_network
  data = ReID_net.valid_data
  output = network.get_output_layer().outputs
  image_input = data.image
  boxes_input = data.boxes
  crop_list_output = data.crop_list
  # ReID_embeddings = ReID_net.session.run([output], feed_dict={image_input: image, boxes_input: boxes})[0][0]
  ReID_embeddings,crop_list = ReID_net.session.run([output,crop_list_output],feed_dict={image_input: image,boxes_input:boxes})
  ReID_embeddings = ReID_embeddings[0]
  for idx,ReID in enumerate(ReID_embeddings):
    proposals[idx]["ReID"] = ReID.tolist()
    # print (ReID.shape)

  # for crop in crop_list:
  #   plt.imshow(crop)
  #   plt.show()

  return(proposals)