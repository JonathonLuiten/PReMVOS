from .Forwarder import Forwarder
import numpy as np
from ReID_net.Log import log
import time
# from ReID_net.Forwarding.CMC_Validator import do_cmc_validation_triplet
# import ReID_net.Measures as Measures

class MARKET1501Forwarder(Forwarder):
  def __init__(self, engine):
    super(MARKET1501Forwarder, self).__init__(engine)
    self.engine = engine

  def forward(self, network, data, save_results=True, save_logits=False):

    file = self.engine.valid_data.file

    output_dir = data.data_dir + "/" + file + "_output.txt"

    outfile = open(output_dir, 'w')

    idx_placeholder = data.idx_placeholder
    batch_size = network.batch_size

    # out_layer = network.tower_layers[0]["outputTriplet"]
    out_layer = network.tower_layers[0]["fc1"]
    assert len(out_layer.outputs) == 1
    out_feature = out_layer.outputs[0]

    features = np.empty([0, 128])

    m = self.engine.valid_data.m
    idx = 0
    while idx < m:
      start = time.time()
      idx_value = [idx, min(idx + batch_size, m), 0, 0]

      feature_val,debug = self.engine.session.run([out_feature, network.tags], feed_dict={idx_placeholder: idx_value})
      # print feature_val.shape
      # features = np.concatenate((features, feature_val), axis=0)
      np.savetxt(outfile, feature_val)
      # print debug

      end = time.time()
      elapsed = end - start
      print(min(idx + batch_size, m), '/', m, "elapsed", elapsed, file=log.v5)

      idx += batch_size

    outfile.close()

    # start = time.time()
    # valid_loss, valid_measures = do_cmc_validation_triplet(self.engine, self.engine.test_network, self.engine.valid_data)
    # end = time.time()
    # elapsed = end - start
    # # train_error_string = Measures.get_error_string(train_measures, "train")
    # valid_error_string = Measures.get_error_string(valid_measures, "valid")
    # print >> log.v1, "epoch", 1, "finished. elapsed:", "%.5f" % elapsed, "valid_score:", valid_loss, valid_error_string

