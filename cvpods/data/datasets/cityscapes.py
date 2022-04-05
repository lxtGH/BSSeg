import copy
import functools
import glob
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
from itertools import chain
import pycocotools.mask as mask_util
from panopticapi.utils import rgb2id
import torch
from PIL import Image

from cvpods.structures import BoxMode
from cvpods.utils import PathManager, comm
from cvpods.data.datasets.target_generator import PanopticDeepLabTargetGenerator

from ..registry import DATASETS
from ..base_dataset import BaseDataset
from ..detection_utils import (
    annotations_to_instances, check_image_size, create_keypoint_hflip_indices,
    filter_empty_instances, read_image
)
from .builtin_meta import _get_builtin_metadata
from .paths_route import _PREDEFINED_SPLITS_CITYSCAPES

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

"""
This file contains functions to parse COCO-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger(__name__)


@DATASETS.register()
class CityScapesDataset(BaseDataset):

    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(CityScapesDataset, self).__init__(cfg, dataset_name, transforms, is_train)
        info = self.name.split("_")
        self.task = info[info.index("seg") - 1]
        self.mode = info[-1]
        assert self.task in ["instance", "sem", "panoptic", "panoptic-deeplab"], "unsupported task {}".format(self.task)

        image_root, json_file = _PREDEFINED_SPLITS_CITYSCAPES["cityscapes"][self.name]
        self.json_file = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root

        self.meta = self._get_metadata()

        self.dataset_dicts = self._load_annotations(
            self.image_root,
            self.json_file,
        )

        if is_train:
            self.dataset_dicts = self._filter_annotations()
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        # fmt: off
        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on

        # panoptic deeplab ground truth loading
        if self.task == "panoptic-deeplab":
            """
                The panoptic_target_generator takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
            """
            self.panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=self.meta["ignore_label"],
            thing_ids=list(self.meta["thing_dataset_id_to_contiguous_id"].values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )
        else:
            self.panoptic_target_generator = None

    def __getitem__(self, index):
        """
        Load data, apply transforms, convert to Instances.
        """
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read image
        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)

        if self.task == "panoptic-deeplab":
            # Panoptic label is encoded in RGB image.
            assert self.data_format == "RGB", "we only support RGB inputs for panoptic segmentation"
            pan_seg_gt = read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            annotations = []
            annotations.append({"sem_seg": pan_seg_gt}) # Hack: using semantic mask annotation for panoptic png
            # apply transfrom
            image, annotations = self._apply_transforms(
                image, annotations)

            # get aug panoptic seg maps
            pan_seg_gt = annotations[0].pop("sem_seg")
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            # Generates training targets for Panoptic-DeepLab.
            targets = self.panoptic_target_generator(rgb2id(pan_seg_gt), dataset_dict["segments_info"])
            dataset_dict.update(targets)

            return dataset_dict
        else:
            if "annotations" in dataset_dict:
                annotations = dataset_dict.pop("annotations")
                annotations = [
                    ann for ann in annotations if ann.get("iscrowd", 0) == 0]
            else:
                annotations = None

            if "sem_seg_file_name" in dataset_dict:
                assert annotations is None
                annotations = []
                with PathManager.open(dataset_dict.get("sem_seg_file_name"), "rb") as f:
                    sem_seg_gt = Image.open(f)
                    sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
                annotations.append({"sem_seg": sem_seg_gt})

            # apply transfrom
            image, annotations = self._apply_transforms(
                image, annotations)

            if "sem_seg_file_name" in dataset_dict:
                dataset_dict.pop("sem_seg_file_name")
                sem_seg_gt = annotations[0].pop("sem_seg")
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
                dataset_dict["sem_seg"] = sem_seg_gt
                annotations = None

            if annotations is not None:
                image_shape = image.shape[:2]  # h, w

                instances = annotations_to_instances(
                    annotations, image_shape, mask_format=self.mask_format
                )

                # # Create a tight bounding box from masks, useful when image is cropped
                # if self.crop_gen and instances.has("gt_masks"):
                #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

                dataset_dict["instances"] = filter_empty_instances(instances)

            # convert to Instance type
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            # h, w, c -> c, h, w
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image.transpose(2, 0, 1)))

            return dataset_dict

    def __reset__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_dicts)

    def _load_annotations(self, image_dir, gt_dir, from_json=True, to_polygons=True):
        if self.task == "instance":
            return self._load_instance_annotations(image_dir, gt_dir, from_json, to_polygons)
        elif self.task == "panoptic-deeplab":
            return self._load_panoptic_annotations(image_dir, gt_dir)
        else:
            return self._load_semantic_annotations(image_dir, gt_dir)

    def _load_instance_annotations(self, image_dir, gt_dir, from_json=True, to_polygons=True):
        """
        Args:
            image_dir (str): path to the raw datasets. e.g., "~/cityscapes/leftImg8bit/train".
            gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
            from_json (bool): whether to read annotations from the raw json file or the png files.
            to_polygons (bool): whether to represent the segmentation as polygons
                (COCO's format) instead of masks (cityscapes's format).

        Returns:
            list[dict]: a list of dicts in cvpods standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )
        """
        if from_json:
            assert to_polygons, (
                "Cityscapes's json annotations are in polygon format. "
                "Converting to mask format is not supported now."
            )
        files = []
        for image_file in glob.glob(os.path.join(image_dir, "**/*.png")):
            suffix = "leftImg8bit.png"
            assert image_file.endswith(suffix)
            prefix = image_dir
            instance_file = (gt_dir + image_file[len(prefix) : -len(suffix)]
                             + "gtFine_instanceIds.png")
            assert os.path.isfile(instance_file), instance_file

            label_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_labelIds.png"
            assert os.path.isfile(label_file), label_file

            json_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_polygons.json"
            files.append((image_file, instance_file, label_file, json_file))
        assert len(files), "No images found in {}".format(image_dir)

        logger = logging.getLogger(__name__)
        logger.info("Preprocessing cityscapes annotations ...")
        # This is still not fast: all workers will execute duplicate works and will
        # take up to 10m on a 8GPU server.
        pool = mp.Pool(processes=max(mp.cpu_count() // comm.get_world_size() // 2, 4))

        ret = pool.map(
            functools.partial(
                cityscapes_files_to_dict,
                from_json=from_json,
                to_polygons=to_polygons
            ),
            files,
        )
        logger.info("Loaded {} images from {}".format(len(ret), image_dir))

        # Map cityscape ids to contiguous ids
        from cityscapesscripts.helpers.labels import labels

        labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
        dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
        for dict_per_image in ret:
            for anno in dict_per_image["annotations"]:
                anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
        return ret

    def _load_semantic_annotations(self, image_dir, gt_dir):
        """
        Args:
            image_dir (str): path to the raw datasets. e.g., "~/cityscapes/leftImg8bit/train".
            gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

        Returns:
            list[dict]: a list of dict, each has "file_name" and
                "sem_seg_file_name".
        """
        ret = []
        for image_file in glob.glob(os.path.join(image_dir, "**/*.png")):
            suffix = "leftImg8bit.png"
            assert image_file.endswith(suffix)
            prefix = image_dir

            label_file = (gt_dir + image_file[len(prefix) : -len(suffix)]
                          + "gtFine_labelTrainIds.png")
            assert os.path.isfile(
                label_file
            ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa

            json_file = gt_dir + image_file[len(prefix) : -len(suffix)] + "gtFine_polygons.json"

            with PathManager.open(json_file, "r") as f:
                jsonobj = json.load(f)
            ret.append(
                {
                    "file_name": image_file,
                    "sem_seg_file_name": label_file,
                    "height": jsonobj["imgHeight"],
                    "width": jsonobj["imgWidth"],
                }
            )
        return ret

    def _load_panoptic_annotations(self, image_dir, gt_dir):
        """
            Load the panoptic and semantic file at the same time usd by the panoptic deeplab
        """
        if self.mode == "train":
            self.gt_raw_annotations = osp.join(gt_dir, "../" ,"cityscapes_panoptic_train")
            self.gt_json = osp.join(gt_dir, "../", "cityscapes_panoptic_train.json")
        elif self.mode == "val":
            self.gt_raw_annotations = osp.join(gt_dir, "../", "cityscapes_panoptic_val")
            self.gt_json = osp.join(gt_dir, "../", "cityscapes_panoptic_val.json")
        else:
            # Do not support test mode now !
            raise NotImplementedError("not such mode")


        # load more meta for evaluation
        self.meta.panoptic_root = self.gt_raw_annotations
        self.meta.panoptic_json = self.gt_json
        self.meta.image_root = image_dir
        self.meta.gt_dir = gt_dir

        return load_cityscapes_panoptic(image_dir, self.gt_raw_annotations, self.gt_json, self.meta)

    def _get_metadata(self):
        meta = _get_builtin_metadata("cityscapes")
        if self.task == "instance":
            meta["evaluator_type"] = "cityscapes_instance_seg"
        elif self.task == "sem":
            meta["evaluator_type"] = "cityscapes_sem_seg"
        elif self.task == "panoptic-deeplab":  # currently we only support panoptic deeplabv3_r50
            meta["evaluator_type"] = "cityscapes_panoptic_seg"

        meta["image_dir"] = self.image_root
        meta["gt_dir"] = self.json_file
        meta["ignore_label"] = 255

        return meta

    def evaluate(self, predictions):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        return self.dataset_dicts


def cityscapes_files_to_dict(files, from_json, to_polygons):
    """
    Parse cityscapes annotation files to a dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in cvpods Dataset format.
    """
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, json_file = files

    annos = []

    if from_json:
        from shapely.geometry import MultiPolygon, Polygon

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }

        # `polygons_union` contains the union of all valid polygons.
        polygons_union = Polygon()

        # CityscapesScripts draw the polygons in sequential order
        # and each polygon *overwrites* existing ones. See
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
        # We use reverse order, and each polygon *avoids* early ones.
        # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

            # Cityscapes's raw annotations uses integer coordinates
            # Therefore +0.5 here
            poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
            # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
            # polygons for evaluation. This function operates in integer space
            # and draws each pixel whose center falls into the polygon.
            # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
            # We therefore dilate the input polygon by 0.5 as our input.
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)

            if not label.hasInstances or label.ignoreInEval:
                # even if we won't store the polygon it still contributes to overlaps resolution
                polygons_union = polygons_union.union(poly)
                continue

            # Take non-overlapping part of the polygon
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)

            anno = {}
            anno["iscrowd"] = label_name.endswith("group")
            anno["category_id"] = label.id

            if isinstance(poly_wo_overlaps, Polygon):
                poly_list = [poly_wo_overlaps]
            elif isinstance(poly_wo_overlaps, MultiPolygon):
                poly_list = poly_wo_overlaps.geoms
            else:
                raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))

            poly_coord = []
            for poly_el in poly_list:
                # COCO API can work only with exterior boundaries now, hence we store only them.
                # TODO: store both exterior and interior boundaries once other parts of the
                # codebase support holes in polygons.
                poly_coord.append(list(chain(*poly_el.exterior.coords)))
            anno["segmentation"] = poly_coord
            (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            annos.append(anno)
    else:
        # See also the official annotation parsing scripts at
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
        with PathManager.open(instance_id_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        flattened_ids = np.unique(inst_image[inst_image >= 24])

        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        for instance_id in flattened_ids:
            # For non-crowd annotations, instance_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            anno = {}
            anno["iscrowd"] = instance_id < 1000
            anno["category_id"] = label.id

            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            if to_polygons:
                # This conversion comes from D4809743 and D5171122,
                # when Mask-RCNN was first developed.
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                    -2
                ]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                # opencv's can produce invalid polygons
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
            annos.append(anno)
    ret["annotations"] = annos
    return ret


def get_cityscapes_panoptic_files(image_dir, gt_dir, json_info):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    image_dict = {}
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = "_leftImg8bit.png"
            assert basename.endswith(suffix), basename
            basename = os.path.basename(basename)[: -len(suffix)]

            image_dict[basename] = image_file

    for ann in json_info["annotations"]:
        image_file = image_dict.get(ann["image_id"], None)
        assert image_file is not None, "No image {} found for annotation {}".format(
            ann["image_id"], ann["file_name"]
        )
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]

        files.append((image_file, label_file, segments_info))

    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0][0]), files[0][0]
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files


def load_cityscapes_panoptic(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw datasets. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    assert os.path.exists(
        gt_json
    ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
    with open(gt_json) as f:
        json_info = json.load(f)
    files = get_cityscapes_panoptic_files(image_dir, gt_dir, json_info)
    ret = []
    for image_file, label_file, segments_info in files:
        sem_label_file = (
            image_file.replace("leftImg8bit", "gtFine").split(".")[0] + "_labelTrainIds.png"
        )
        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
                ),
                "sem_seg_file_name": sem_label_file,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    assert PathManager.isfile(
        ret[0]["pan_seg_file_name"]
    ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    return ret
