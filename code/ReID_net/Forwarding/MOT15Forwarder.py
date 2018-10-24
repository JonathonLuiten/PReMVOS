from .Forwarder import Forwarder
import numpy
from ReID_net.Log import log
import time

class MOT15Forwarder(Forwarder):
  def __init__(self, engine):
    super(MOT15Forwarder, self).__init__(engine)
    self.data = engine.valid_data.seq_data.eval()
    self.engine = engine

  def forward(self, network, data, save_results=True, save_logits=False):
    for seq_num in range(data.num_seq):

      T = data.Ts[seq_num]
      self.window_size = 10
      # self.window_size = T

      seq = data.seq_list[seq_num]
      output_dir = data.data_dir + "/" + seq + "/det/comparison_triplet_features.txt"
      outfile = open(output_dir, 'w')
      features = self._extract_intermediate_features(network, data, seq_num)
      # self._compare_features(self.engine, network, data, features,outfile, seq_num)
      self._save_intermediate_features(self.engine, network, data, features,outfile, seq_num)

  def _extract_intermediate_features(self, network, data, seq_num):
  # def _extract_intermediate_features(self, network, data):

    idx_placeholder = data.idx_placeholder
    batch_size = network.batch_size
    # seq_data = data.seq_data.eval(feed_dict={data.seq_num_placeholder:seq_num})
    # seq_data = data.seq_data

    out_layer = network.tower_layers[0]["fc1"]
    assert len(out_layer.outputs) == 1
    out_feature = out_layer.outputs[0]
    out_feature_size = out_layer.n_features

    features = numpy.empty([0, out_feature_size])

    # m = seq_data.shape[0]
    m = data.Ms[seq_num]
    idx = 0
    while idx < m:
      start = time.time()
      idx_value = [idx, min(idx + 2 * batch_size, m), 0, 1]

      feature_val = self.engine.session.run([out_feature], feed_dict={idx_placeholder: idx_value, data.seq_num_placeholder:seq_num})
      # feature_val = self.engine.session.run([out_feature],feed_dict={idx_placeholder: idx_value})
      features = numpy.concatenate((features, feature_val[0]), axis=0)

      end = time.time()
      elapsed = end - start
      print(min(idx + 2 * batch_size, m), '/', m, "elapsed", elapsed, file=log.v5)
      idx += 2 * batch_size

    return features

  def _compare_features(self, engine, network, data, features,outfile, seq_num):
  # def _compare_features(self, engine, network, data, features, outfile):
    y = network.y_softmax
    in_layer = network.tower_layers[0]["siam_concat"]
    assert len(in_layer.outputs) == 1
    in_feature = in_layer.outputs[0]

    merge_type = engine.config.str("merge_type", "")

    start = time.time()
    # seq_data = data.seq_data.eval(feed_dict={data.seq_num_placeholder: seq_num})
    # seq_data = data.seq_data
    # m = seq_data.shape[0]
    m = data.Ms[seq_num]
    # T = int(seq_data[:, 0].max())
    window_size = self.window_size #T #10
    inc = numpy.arange(m)
    seq_data = self.data[data.look_up[seq_num]:data.look_up[seq_num+1],:]

    for idx1 in range(m):
      t = seq_data[idx1, 0]
      t = t.astype(int)
      for future_t in range(t+1,t+window_size+1):
        idx2 = inc[seq_data[:, 0] == future_t]
        feature1 = features[(idx1,) * idx2.size, :]
        feature2 = features[idx2,:]

        if merge_type == "add":
          feature_val = feature1 + feature2
        elif merge_type == "subtract":
          feature_val = feature1 - feature2
        elif merge_type == "abs_subtract":
          feature_val = numpy.abs(feature1 - feature2)
        else:  # merge_type == "concat":
          feature_val = numpy.concatenate((feature1, feature2), axis=1)

        y_val = engine.session.run(y, feed_dict={in_feature: feature_val})
        for i in range(idx2.size):
          outfile.write("%i %i %f\n"%(idx1,idx2[i],y_val[i][1]))

    end = time.time()
    elapsed = end - start
    print("elapsed", elapsed, file=log.v5)
    print(file=log.v4)

  def _save_intermediate_features(self, engine, network, data, features, outfile, seq_num):
    import pickle as pickle
    pickle.dump(features,outfile)
    # for idx in range(len(features)):
    #   outfile.write("%i %f\n" % (idx,  features[idx,:]))
