import tensorflow as tf

from ReID_net.Forwarding.Forwarder import Forwarder
from ReID_net.datasets.Util.Timer import Timer
from ReID_net.Log import log


class DatasetSpeedtestForwarder(Forwarder):
  def __init__(self, engine):
    super(DatasetSpeedtestForwarder, self).__init__(engine)

  def forward(self, network, data, save_results=True, save_logits=False):
    batch_size = self.config.int("batch_size", -1)
    assert batch_size != -1
    values = list(data.create_input_tensors_dict(batch_size).values())
    tf.train.start_queue_runners(self.session)
    n_runs = 2000
    print("batch_size", batch_size, file=log.v1)
    with Timer("%s runs tooks" % n_runs):
      for idx in range(n_runs):
        print(idx, "/", n_runs, file=log.v5)
        self.session.run(values)
