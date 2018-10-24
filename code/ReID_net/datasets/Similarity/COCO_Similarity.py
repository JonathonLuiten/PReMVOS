from .Similarity import SimilarityDataset
from ReID_net.datasets.COCO.COCO import COCODataset


class COCOSimilarityDataset(SimilarityDataset):
  def __init__(self, config, subset, coord):
    self.coco_dataset = COCODataset(config, subset, coord)
    ann_id_to_filename = {}
    for filename, anns in list(self.coco_dataset.filename_to_anns.items()):
      for ann in anns:
        ann_id_to_filename[ann["id"]] = filename
    annotations = self.coco_dataset.anns
    for ann in annotations:
      file_name = ann_id_to_filename[ann["id"]]
      data_type = "train2014" if "train2014" in file_name else "val2014"
      img_dir = '%s/%s/' % (self.coco_dataset.data_dir, data_type)
      ann["img_file"] = img_dir + file_name
    cat_ids = self.coco_dataset.coco.getCatIds()
    for ann in annotations:
      ann["category_id"] = cat_ids.index(ann["category_id"]) + 1
    super(COCOSimilarityDataset, self).__init__(config, subset, coord, annotations, n_train_ids=80)
