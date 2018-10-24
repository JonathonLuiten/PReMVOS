import tensorflow as tf

from ReID_net.datasets.DAVIS.DAVIS import DAVIS_DEFAULT_PATH, DAVIS_FLOW_DEFAULT_PATH,  NUM_CLASSES, DAVIS_IMAGE_SIZE, \
  VOID_LABEL, read_image_and_annotation_list, group_into_sequences, get_input_list_file
from ReID_net.datasets.Dataset import ImageDataset
from ReID_net.datasets.Util.Reader import load_image_tensorflow, load_png_mask_tensorflow, \
  load_flow_from_flo_tensorflow
from ReID_net.datasets.Util.Input import assemble_input_tensors
from ReID_net.datasets.Util.Reader import create_tensor_dict
from ReID_net.datasets.Util.MaskDamager import damage_mask
from ReID_net.datasets.Util.Resize import resize


class DAVISMaskTransferDataset(ImageDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(DAVISMaskTransferDataset, self).__init__("davis_mask", DAVIS_DEFAULT_PATH, NUM_CLASSES, config, subset, coord,
                                                   DAVIS_IMAGE_SIZE, VOID_LABEL, fraction)
    self.flow_dir = config.str("davis_flow_data_dir", DAVIS_FLOW_DEFAULT_PATH)
    self.flow_into_past = config.bool("flow_into_past", False)
    self.flow_into_future = config.bool("flow_into_future", False)
    self.flow_as_angle = config.bool("flow_as_angle", False)

    self.old_mask_scale_factor = self.config.float("old_mask_scale_factor", 0.0)
    self.old_mask_shift_factor = self.config.float("old_mask_shift_factor", 0.0)
    self.old_mask_shift_absolute = self.config.float("old_mask_shift_absolute", 0.0)

    self.trainsplit = config.int("trainsplit", 0)

  def read_inputfile_lists(self):
    assert self.subset in ("train", "valid"), self.subset
    list_file = get_input_list_file(self.subset, self.trainsplit)
    imgs, labels = read_image_and_annotation_list(self.data_dir + list_file, self.data_dir)

    imgs = group_into_sequences(imgs)
    labels = group_into_sequences(labels)
    old_labels = [x[:-1] for x in labels]

    # filter out first frames, since we have no old masks and no flow
    imgs = [x[1:] for x in imgs]
    labels = [x[1:] for x in labels]

    # if we use flow_into_future also filter out last frames
    if self.flow_into_future:
      imgs = [x[:-1] for x in imgs]
      labels = [x[:-1] for x in labels]
      old_labels = [x[:-1] for x in old_labels]

    #flatten lists of lists
    imgs = sum(imgs, [])
    labels = sum(labels, [])
    old_labels = sum(old_labels, [])

    flows = []
    if self.flow_into_past:
      flow_past = [self.flow_dir + "Flow_backward/" + x[x.index("480p"):].replace(".jpg", ".flo") for x in imgs]
      flows.append(flow_past)
    if self.flow_into_future:
      flow_future = [self.flow_dir + "Flow_forward/" + x[x.index("480p"):].replace(".jpg", ".flo") for x in imgs]
      flows.append(flow_future)

    assert len(imgs) == len(labels) == len(old_labels)
    assert all([len(f) == len(imgs) for f in flows])

    return [imgs, labels, old_labels] + flows

  def _read_inputfiles(self, queue, resize_mode, input_size, augmentors):
    im_path = queue[0]
    label_path = queue[1]
    old_label_path = queue[2]

    img = load_image_tensorflow(im_path, jpg=True)
    old_label = load_png_mask_tensorflow(old_label_path)

    def my_damage_mask(mask):
      return damage_mask(mask, self.old_mask_scale_factor, self.old_mask_shift_absolute, self.old_mask_shift_factor)
    old_label, = tf.py_func(my_damage_mask, [old_label], [tf.float32])

    label = load_png_mask_tensorflow(label_path)

    flow_past = flow_future = None
    if self.flow_into_past:
      flow_past_path = queue[3]
      flow_past = load_flow_from_flo_tensorflow(flow_past_path, self.flow_as_angle)
    if self.flow_into_future:
      flow_future_path = queue[4] if self.flow_into_past else queue[3]
      flow_future = load_flow_from_flo_tensorflow(flow_future_path, self.flow_as_angle)

    tensors = create_tensor_dict(unnormalized_img=img, label=label, tag=im_path, raw_label=label, old_label=old_label,
                                 flow_past=flow_past, flow_future=flow_future)

    resize_mode, input_size = self._get_resize_params(self.subset, self.image_size)
    tensors = resize(tensors, resize_mode, input_size)

    for augmentor in augmentors:
      tensors = augmentor.apply(tensors)

    tensors = assemble_input_tensors(tensors, DAVIS_IMAGE_SIZE)
    label.set_shape(list(DAVIS_IMAGE_SIZE) + [1])

    summaries = []

    return tensors, summaries
