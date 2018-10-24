import time


class Timer(object):
  def __init__(self, message="", stream=None):
    if stream is None:
      from ReID_net.Log import log
      stream = log.v4
    self.stream = stream
    self.start = None
    self.message = message

  def __enter__(self):
    self.start = time.time()

  def __exit__(self, exc_type, exc_val, exc_tb):
    end = time.time()
    start = self.start
    self.start = None
    elapsed = end - start
    print(self.message, "elapsed", elapsed, file=self.stream)
