from ReID_net.datasets.Dataset import ImageDataset
import tensorflow as tf


def zero_label(img_path, label_path):
  #TODO: we load the image again just to get it's size which is kind of a waste (but should not matter for most cases)
  img_contents = tf.read_file(img_path)
  img = tf.image.decode_image(img_contents, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img.set_shape((None, None, 3))
  label = tf.zeros_like(img, dtype=tf.uint8)[..., 0:1]
  res = {"label": label}
  return res


class CustomDataset(ImageDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(CustomDataset, self).__init__("custom", "", 2, config, subset, coord, None, 255, fraction,
                                        label_load_fn=zero_label)
    self.file_list = config.str("file_list")

  def read_inputfile_lists(self):
    imgs = [x.strip() for x in open(self.file_list).readlines()]
    labels = ["" for _ in imgs]
    return [imgs, labels]
