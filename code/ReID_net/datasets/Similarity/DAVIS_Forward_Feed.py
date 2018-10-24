import tensorflow as tf
from ReID_net.datasets.Dataset import Dataset
from ReID_net.datasets.Util.Normalization import normalize
from ReID_net.datasets.Util.Resize import resize_image

DEFAULT_INPUT_SIZE = [128, 128]

class DAVISForwardFeedDataset(Dataset):
  def __init__(self, config, subset, coord):
    super(DAVISForwardFeedDataset, self).__init__(subset)
    assert subset in ("train", "valid"), subset
    self.config = config
    self.subset = subset
    self.coord = coord
    self.model = config.str("model", "")

    self.input_size = config.int_list("input_size", DEFAULT_INPUT_SIZE)
    self.input_size = tuple(self.input_size)
    self.group_size = config.int("group_size", 2)
    self.pair_ratio = config.int("pair_ratio", 1)
    self.context_region_factor = config.float("context_region_factor", 1.2)

    self.n_classes = config.int("num_classes", None)

    # self.image = tf.placeholder(tf.float32, (None,None,3))
    self.image = tf.placeholder(tf.float32, (None, None, 3))
    self.conv_image = self.image/255
    # self.image = tf.placeholder(tf.int32, (None, None, 3))
    # self.conv_image = tf.image.convert_image_dtype(self.image, tf.float32)
    self.boxes = tf.placeholder(tf.float32,(None,4))

  def num_classes(self):
    return self.n_classes

  def apply_contex_region(self,boxes,dims):
    xs = boxes[:, 0]
    ys = boxes[:, 1]
    ws = boxes[:, 2]
    hs = boxes[:, 3]
    # add context region
    xs -= 0.5 * ws * (self.context_region_factor - 1.0)
    ys -= 0.5 * hs * (self.context_region_factor - 1.0)
    ws *= self.context_region_factor
    hs *= self.context_region_factor
    # round to integer coordinates
    xs = tf.cast(tf.round(xs), tf.int32)
    ys = tf.cast(tf.round(ys), tf.int32)
    ws = tf.cast(tf.round(ws), tf.int32)
    hs = tf.cast(tf.round(hs), tf.int32)
    # clip to image size
    xs = tf.maximum(xs, 0)
    ys = tf.maximum(ys, 0)
    excess_ws = tf.maximum(xs + ws - dims[1], 1)
    excess_hs = tf.maximum(ys + hs - dims[0], 1)
    ws = ws - excess_ws
    hs = hs - excess_hs

    boxes = tf.cast(tf.stack([xs, ys, ws, hs], 1), tf.float32)

    # # convert to normalised tf.image coords
    # x1s = xs / (dims[1] - 1)
    # x2s = (xs+ws) / (dims[1] - 1)
    # y1s = ys / (dims[0] - 1)
    # y2s = (ys+hs) / (dims[0] - 1)
    # boxes = tf.cast(tf.stack([y1s, x1s, y2s, x2s],1),tf.float32)
    return boxes

  def do_one_crop(self,box):
    # Do the crop and resize myself
    x = tf.cast(box[0], tf.int32)
    y = tf.cast(box[1], tf.int32)
    w = tf.cast(box[2], tf.int32)
    h = tf.cast(box[3], tf.int32)
    # image = tf.expand_dims(self.image, 0)
    img_cropped = self.conv_image[y:y + h, x:x + w]
    min_dim = tf.minimum(h,w)

    # # resize
    # img = resize_image(img_cropped, self.input_size, True)

    # resize
    img = tf.cond(min_dim>10,lambda: resize_image(img_cropped, self.input_size, True),
                  lambda: tf.zeros([self.input_size[0],self.input_size[1],3]))


    # img.set_shape(self.input_size + (3,))
    norm_img = normalize(img)
    return norm_img
    # return img

  def _create_inputs_for_eval(self, batch_size):

    dims = tf.shape(self.image)
    # image = tf.expand_dims(self.image,0)
    boxes = self.apply_contex_region(self.boxes,dims)

    # box_ind = tf.fill([tf.shape(boxes)[0],],0)
    # imgs = tf.image.crop_and_resize(image, boxes, box_ind, self.input_size)

    imgs = tf.map_fn(self.do_one_crop,boxes)

    # orig_imgs = imgs
    # imgs = normalize(imgs)
    return imgs

  def create_input_tensors_dict(self, batch_size):
    imgs = self._create_inputs_for_eval(batch_size)
    one_labels = tf.fill([tf.shape(imgs)[0]],1)

    # for debugging
    self.crop_list = imgs

    # tensors = {"inputs": imgs, "labels": labels, "tags": tags, "original_labels": original_classes}
    tensors = {"inputs": imgs, "labels":one_labels,"original_labels":one_labels}
    return tensors

  def num_examples_per_epoch(self):
    return 256

  def void_label(self):
    return None