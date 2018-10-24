from ReID_net.Forwarding.Forwarder import ImageForwarder
from ReID_net.Log import log
from ReID_net.Measures import average_measures


class InteractiveImageForwarder(ImageForwarder):
  def __init__(self, engine):
    super(InteractiveImageForwarder, self).__init__(engine)
    #hope moving this import here, does not break anything, otherwise move it back up again
    import matplotlib
    matplotlib.use('Agg')

  def forward(self, network, data, save_results=True, save_logits=False):
    n_total = data.num_examples_per_epoch()
    n_processed = 0
    targets = network.raw_labels
    ys = network.ys_resized

    # e.g. used for resizing
    # ys = self._adjust_results_to_targets(ys, targets)

    measures = []
    ys_argmax_values = []
    while n_processed < n_total:
      n, new_measures, ys_argmax_val, _, _ = self._process_forward_minibatch(data, network, save_logits, save_results,
                                                                             targets, ys, n_processed)
      measures += new_measures
      ys_argmax_values += list(ys_argmax_val)
      n_processed += n
      print(n_processed, "/", n_total, file=log.v5)
    if self.ignore_first_and_last_results:
      measures = measures[1:-1]
    elif self.ignore_first_result:
      measures = measures[1:]

    measures = average_measures(measures)
    if hasattr(data, "video_tag"):
      video_idx = data.get_video_idx()
      print("sequence", video_idx + 1, data.video_tag(video_idx), measures, file=log.v1)
    else:
      print(measures, file=log.v1)

    return ys_argmax_values, measures
