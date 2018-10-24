import numpy
import numpy.matlib
import time
from ReID_net.Log import log

def do_cmc_validation(engine,network,data):
  m = data.num_test_id
  n = m * m
  idx_placeholder = data.idx_placeholder
  batch_size = network.batch_size
  debug = network.tags
  path = data.test_case
  end_net = data.use_end_network
  rank_out = ""
  errors = ""
  measures = {}
  merge_type = engine.config.str("merge_type", "")

  out_layer_name = engine.config.str("output_embedding_layer","fc1")
  out_layer = network.tower_layers[0][out_layer_name]
  assert len(out_layer.outputs) == 1
  out_feature = out_layer.outputs[0]
  out_feature_size = out_layer.n_features

  test_cases = engine.config.unicode_list("test_cases", [])

  for test_case in test_cases:
    errs = 0
    y_vals = numpy.empty([0,1])
    probe = numpy.empty([0, out_feature_size])
    gallery = numpy.empty([0, out_feature_size])

    idx = 0
    while idx < m:
      start = time.time()
      idx_value = [idx, min(idx + batch_size, m),1,0]

      feature_val, msg = engine.session.run([out_feature, debug],
                                            feed_dict={idx_placeholder: idx_value, path: test_case, end_net: False})
      probe = numpy.concatenate((probe, feature_val), axis=0)

      end = time.time()
      elapsed = end - start
      print(min(idx + batch_size, m), '/', m, "elapsed", elapsed, file=log.v5)
      idx += batch_size

    idx = 0
    while idx < m:
      start = time.time()
      idx_value = [idx, min(idx + batch_size, m), 1, 1]

      feature_val, msg = engine.session.run([out_feature, debug],
                                            feed_dict={idx_placeholder: idx_value, path: test_case, end_net: False})
      gallery = numpy.concatenate((gallery, feature_val), axis=0)

      end = time.time()
      elapsed = end - start
      print(min(idx + batch_size, m), '/', m, "elapsed", elapsed, file=log.v5)
      idx += batch_size

    start = time.time()
    for pdx in range(m):
      idx = 0
      while idx < m:
        idx_value = [idx, min(idx + batch_size, m), pdx, 1]
        r = numpy.arange(idx_value[0], idx_value[1])
        q = (pdx,) * (min(idx + batch_size, m) - idx)

        if data.validation_mode == "similarity":

          y = network.y_softmax
          e = network.measures_accumulated
          in_layer_name = engine.config.str("input_embedding_layer", "siam_concat")
          in_layer = network.tower_layers[0][in_layer_name]
          assert len(in_layer.outputs) == 1
          in_feature = in_layer.outputs[0]

          if merge_type == "add":
            feature_val = probe[q, :] + gallery[r, :]
          elif merge_type == "subtract":
            feature_val = probe[q, :] - gallery[r, :]
          elif merge_type == "abs_subtract":
            feature_val = numpy.abs(probe[q, :] - gallery[r, :])
          else: # merge_type == "concat":
            feature_val = numpy.concatenate((probe[q, :], gallery[r, :]), axis=1)

          y_val, err = engine.session.run([y, e], feed_dict={idx_placeholder: idx_value, in_feature: feature_val, end_net: True,path: test_case})
          y_val = y_val[:,0:1]
          errs += err["errors"]

        else: # data.validation_mode == "embedding":
          y_val = numpy.linalg.norm(probe[q,:] - gallery[r,:],axis=1)
          y_val = numpy.reshape(y_val,[y_val.size,1])

        y_vals = numpy.concatenate((y_vals, y_val), axis=0)
        idx += batch_size

    y_vals1 = y_vals
    Apsum = 0
    ranks = numpy.zeros(m)
    for i in range(m):
      r = numpy.arange(m * i, m * (i + 1))
      I = numpy.identity(m)
      corr = I[:, i]
      tab = numpy.column_stack((y_vals1[r], corr))
      id = numpy.argsort(y_vals1[r], axis=0)
      tab = tab[id, :]
      pos = numpy.where(tab[:,0, 1])[0]
      ranks[i] = pos[0] + 1
      Ap = numpy.zeros(1)
      f = numpy.zeros(1)
      for j in range(pos.size):
        f += 1
        Ap += f / (pos[j] + 1)

      Apsum += Ap

    mAp = Apsum / m
    cmc = numpy.zeros(m)
    for i in range(m):
      cmc[i] = 100 / m * ranks[ranks <= i + 1].size

    rank1 = cmc[0]
    rank5 = cmc[4]
    rank10 = cmc[9]
    error = errs / n

    errors += "%.3f " % mAp
    rank_out += "%.1f " % rank1 + "%.1f " % rank5 + "%.1f / " % rank10

    measures = {}
    measures["ranks"] = rank_out

    end = time.time()
    elapsed = end - start
    print(test_case, "elapsed", elapsed, file=log.v5)

  return errors, measures