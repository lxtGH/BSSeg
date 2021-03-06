# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle

from cvpods.utils import PathManager, comm

from .model_loading import align_and_update_state_dicts
from .checkpoint import Checkpointer


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & cvpods
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", resume=False, *, save_to_disk=None, replace_substr_dict={},
                 backbone_prefix="",
                 load_emb_pred_from=None,
                 # load_classifier=True,
                 **checkpointables):
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            resume (bool): indicate whether to resume from latest checkpoint or start from scratch.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the `state_dict()` and `load_state_dict()` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        if len(backbone_prefix) > 0:
            replace_substr_dict[backbone_prefix] = ''
        if load_emb_pred_from is not None:
            replace_substr_dict[f'mmss_heads.{load_emb_pred_from}.v2l_projection.weight'
                               ] = 'roi_heads.box_predictor.emb_pred.weight'
            replace_substr_dict[f'mmss_heads.{load_emb_pred_from}.v2l_projection.bias'
            ] = 'roi_heads.box_predictor.emb_pred.bias'
        # if not load_classifier:
        #     replace_substr_dict[f'predictor.cls_score'] = 'predictor.DONT_LOAD.cls_score'

        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            resume,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        self.replace_substr_dict = replace_substr_dict

    def _load_file(self, filename):
        """
        Args:
            filename (str): load checkpoint file from local. checkpoint can be of type
                pkl, pth
        """
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in cvpods model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        elif filename.endswith(".pth"):
            loaded = super()._load_file(filename)  # load native pth checkpoint
            if "model" not in loaded:
                loaded = {"model": loaded}
            return loaded

    def _load_model(self, checkpoint, replace_substr_dict={}):
        """
        Args:
            checkpoint (dict): model state dict.
        """
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        super()._load_model(checkpoint, self.replace_substr_dict)
