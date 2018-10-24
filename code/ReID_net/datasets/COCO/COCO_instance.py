import tensorflow as tf

from ReID_net.datasets.COCO.COCO import COCODataset
from ReID_net.datasets.Util import Reader


class COCOInstanceDataset(COCODataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(COCOInstanceDataset, self).__init__(config, subset, coord, fraction=fraction)

  def build_filename_to_anns_dict(self):
    for ann in self.anns:
      ann_id = ann['id']
      img_id = ann['image_id']
      img = self.coco.loadImgs(img_id)
      file_name = img[0]['file_name']

      if not ann['iscrowd'] and \
              'segmentation' in ann and \
              'bbox' in ann and ann['area'] > 200:
        file_name = file_name + ":" + repr(ann_id)
        if file_name in self.filename_to_anns:
          print("Ignoring instance as an instance with the same id exists in filename_to_anns.")
          # self.filename_to_anns[file_name].append(ann)
        else:
          self.filename_to_anns[file_name] = [ann]
        # self.filename_to_anns[file_name] = ann

  def img_load_fn(self, img_path):
    path = tf.string_split([img_path], ':').values[0]
    path = tf.string_split([path], '/').values[-1]
    img_dir = tf.cond(tf.equal(tf.string_split([path], '_').values[1], tf.constant("train2014")),
                      lambda: '%s/%s/' % (self.data_dir, "train2014"),
                      lambda: '%s/%s/' % (self.data_dir, "val2014"))
    path = img_dir + path
    return Reader.load_img_default(path)
