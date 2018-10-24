from .Similarity import SimilarityDataset
import glob
from ReID_net.datasets.Util.Reader import load_image_tensorflow
from ReID_net.datasets.Augmentors import apply_augmentors
from ReID_net.datasets.Util.Normalization import normalize

DEFAULT_REID_DATA = '/home/luiten/vision/PReMVOS/data/Re-ID-data'
DEFAULT_NUM_IDS = 242

class DAVISSimilarityDataset(SimilarityDataset):
  def __init__(self, config, subset, coord):

    ReId_data = config.str('reid_input_dir',DEFAULT_REID_DATA)
    num_ids = config.int('num_reid_ids', DEFAULT_NUM_IDS)
    import time
    t = time.time()
    print("starting something else weirdly slow")
    filenames = glob.glob(ReId_data + '/*')
    print("finishing something else weirdly slow",time.time()-t)
    cat_ids = [int(f.split('/')[-1].split('-')[0])+1 for f in filenames]
    annotations = [{"img_file":f,"category_id":c,"bbox":0} for f,c in zip(filenames,cat_ids)]
    super(DAVISSimilarityDataset, self).__init__(config, subset, coord, annotations, n_train_ids=num_ids)

  def _load_crop_helper(self, img_file_name, img_bbox):
    img = load_image_tensorflow(img_file_name, jpg=self.jpg, channels=3)

    img.set_shape(self.input_size + (3,))
    # augment and normalize
    tensors = {"unnormalized_img": img}
    tensors = apply_augmentors(tensors, self.augmentors)
    img = tensors["unnormalized_img"]
    img_norm = normalize(img)
    return img_norm, img, None
