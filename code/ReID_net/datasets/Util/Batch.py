import tensorflow as tf


def create_batch_dict(batch_size, tensors_dict):
  if batch_size == 1:
    batch = {k: tf.expand_dims(t, axis=0) for k, t in list(tensors_dict.items())}
    summary = None
  else:
    keys = list(tensors_dict.keys())
    values = list(tensors_dict.values())
    values = tf.train.batch(values, batch_size, num_threads=8, capacity=5 * batch_size)
    batch = dict(list(zip(keys, values)))
    summary = tf.get_collection(tf.GraphKeys.SUMMARIES)[-1]
    assert "fraction_of_" in summary.name
  for t in list(batch.values()):
    t.set_shape([batch_size] + [None] * (t.get_shape().ndims - 1))
  return batch, summary
