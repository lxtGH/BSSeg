#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File               :   transform_gen.py
@Time               :   2020/05/07 23:50:21
@Author             :   Facebook, Inc. and its affiliates.
@Contact            :   poodarchu@gmail.com
@Last Modified by   :   Benjin Zhu
@Last Modified time :   2020/05/07 23:50:21
'''

import inspect
import pprint
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from PIL import Image

from cvpods.structures import Boxes, BoxMode, pairwise_iou

from ..registry import TRANSFORMS
from .transform import (  # isort:skip
    ScaleTransform,
    AffineTransform,
    BlendTransform,
    IoUCropTransform,
    CropTransform,
    CropPadTransform,
    HFlipTransform,
    HFlipTransformVideo,
    NoOpTransform,
    NoOpTransformVideo,
    VFlipTransform,
    VFlipTransformVideo,
    DistortTransform,
    BoxJitterTransform,
    RandomSwapChannelsTransform,
    ExpandTransform,
    ExtentTransform,
    ResizeTransform,
    ResizeTransformVideo,
    # Transforms used in ssl
    GaussianBlurTransform,
    LightningTransform,
    ComposeTransform,
    TorchTransform,
    # LabSpaceTransform,
    PadTransform,
    JigsawCropTransform,
    EfficientDetResizeCropTransform
)

__all__ = [
    "Pad",
    "RandomScale",
    "Expand",
    "MinIoURandomCrop",
    "RandomSwapChannels",
    "CenterAffine",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomCropWithInstance",
    "RandomCropWithMaxAreaLimit",
    "RandomCropPad",
    "RandomExtent",
    "RandomFlip",
    "RandomFlipVideo",
    "RandomSaturation",
    "RandomLighting",
    "RandomDistortion",
    "RandomBoxJitter",
    "Resize",
    "ResizeVideo",
    "ResizeLongestShortestEdgeVideo",
    "ResizeShortestEdge",
    "ResizeLongestEdge",
    "ShuffleList",
    "RandomList",
    "RepeatList",
    "TransformGen",
    "TorchTransformGen",
    "ComposeListWithProp",
    # transforms used in ssl
    "GaussianBlur",
    "Lightning",
    "JigsawCrop",
]


def check_dtype(img):
    """
    Check the image data type and dimensions to ensure that transforms can be applied on it.

    Args:
        img (np.array): image to be checked.
    """
    assert isinstance(
        img, np.ndarray
    ), "[TransformGen] Needs an numpy array, but got a {}!".format(type(img))
    assert not isinstance(img.dtype, np.integer) or (
        img.dtype == np.uint8
    ), "[TransformGen] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


@TRANSFORMS.register()
class TransformGen(metaclass=ABCMeta):
    """
    TransformGen takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `TransformGen` can be applied with :func:`apply_transform_gens`.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img, annotations=None):
        pass

    def __call__(self, img, annotations=None):
        return self.get_transform(img, annotations)(img, annotations)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGen(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and
                    param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(
                        name
                    )
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


@TRANSFORMS.register()
class RandomFlip(TransformGen):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()

@TRANSFORMS.register()
class RandomFlipVideo(TransformGen):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, imgs, annotationss=None):
        h, w = imgs[0].shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransformVideo(w)
            elif self.vertical:
                return VFlipTransformVideo(h)
        else:
            return NoOpTransformVideo()


@TRANSFORMS.register()
class TorchTransformGen(TransformGen):
    """
    Wrapper transfrom of transforms in torchvision.
    It convert img (np.ndarray) to PIL image, and convert back to np.ndarray after transform.
    """
    def __init__(self, tfm):
        super().__init__()
        self.tfm = tfm

    def get_transform(self, img, annotations=None):
        return TorchTransform(self.tfm)


@TRANSFORMS.register()
class RandomDistortion(TransformGen):
    """
    Random distort image's hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure, image_format="BGR"):
        """
        RandomDistortion Initialization.
        Args:
            hue (float): value of hue
            saturation (float): value of saturation
            exposure (float): value of exposure
        """
        assert image_format in ["RGB", "BGR"]
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return DistortTransform(self.hue, self.saturation, self.exposure, self.image_format)


@TRANSFORMS.register()
class CenterAffine(TransformGen):
    """
    Affine Transform for CenterNet
    """

    def __init__(self, boarder, output_size, pad_value=[0, 0, 0], random_aug=True):
        """
        output_size (w, h) shape
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        img_shape = img.shape[:2]
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)
        return AffineTransform(src, dst, self.output_size, self.pad_value)

    @staticmethod
    def _get_boarder(boarder, size):
        """
        This func may be rewirite someday
        """
        i = 1
        size //= 2
        while size <= boarder // i:
            i *= 2
        return boarder // i

    def generate_center_and_scale(self, img_shape):
        """
        generate center
        shpae : (h, w)
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        if self.random_aug:
            scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
            h_boarder = self._get_boarder(self.boarder, height)
            w_boarder = self._get_boarder(self.boarder, width)
            center[0] = np.random.randint(low=w_boarder, high=width - w_boarder)
            center[1] = np.random.randint(low=h_boarder, high=height - h_boarder)
        else:
            pass

        return center, scale

    @staticmethod
    def generate_src_and_dst(center, scale, output_size):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = scale[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst


@TRANSFORMS.register()
class RandomBoxJitter(TransformGen):
    """
    Random jitter bounding box without changing image.
    """

    def __init__(self, p: float = 0.0, ratio: int = 0):
        """
        Args:
            p (float): probability of performing jitter
            ratio (int): offsets along x and y axis in pixels.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return BoxJitterTransform(self.p, self.ratio)


@TRANSFORMS.register()
class GaussianBlur(TransformGen):
    """
    Gaussian blur transform.
    """
    def __init__(self, sigma, p=1.0):
        """
        Args:
            sigma (List(float)): sigma of gaussian
            p (float): probability of perform this augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return GaussianBlurTransform(self.sigma, self.p)


@TRANSFORMS.register()
class Resize(TransformGen):
    """
    Resize image to a target size
    """

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return ResizeTransform(
            img.shape[0], img.shape[1], self.shape[0], self.shape[1], self.interp
        )


@TRANSFORMS.register()
class ResizeVideo(TransformGen):
    """
    Resize image to a target size
    """

    def __init__(self, shape, keep_ratio=True, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, imgs, annotationss=None):
        if self.keep_ratio:
            h, w = imgs[0].shape[:2]
            scale_factor = min(min(self.shape) / min(h, w),
                               max(self.shape) / max(h, w))
            new_h = int(h * float(scale_factor) + 0.5)
            new_w = int(w * float(scale_factor) + 0.5)
        else:
            new_h = self.shape[0]
            new_w = self.shape[1]
        return ResizeTransformVideo(
            imgs[0].shape[0], imgs[0].shape[1], new_h, new_w, self.interp
        )

@TRANSFORMS.register()
class ResizeLongestShortestEdgeVideo(TransformGen):
    """
    Resize image to a target size
    """

    def __init__(self, img_scales, sample_style="range", keep_ratio=True, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
        """
        super(ResizeLongestShortestEdgeVideo, self).__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        self.num_scales = len(img_scales)
        self._init(locals())

    def get_transform(self, imgs, annotationss=None):
        h, w = imgs[0].shape[:2]
        if self.num_scales == 1:
            img_scale = self.img_scales[0]
            if self.keep_ratio:
                scale_factor = min(min(img_scale) / min(h, w),
                                   max(img_scale) / max(h, w))
                new_h = int(h * float(scale_factor) + 0.5)
                new_w = int(w * float(scale_factor) + 0.5)
            else:
                new_h = int(img_scale[0])
                new_w = int(img_scale[1])
        elif self.num_scales == 2:
            if self.sample_style == "range":
                if self.keep_ratio:
                    img_scale_long = [max(s) for s in self.img_scales]
                    img_scale_short = [min(s) for s in self.img_scales]
                    long_edge = np.random.randint(
                        min(img_scale_long),
                        max(img_scale_long) + 1
                    )
                    short_edge = np.random.randint(
                        min(img_scale_short),
                        max(img_scale_short) + 1
                    )
                    scale_factor = min(short_edge / min(h, w),
                                       long_edge / max(h, w))
                    new_h = int(h * float(scale_factor) + 0.5)
                    new_w = int(w * float(scale_factor) + 0.5)
                else:
                    img_scale_h = [s[0] for s in self.img_scales]
                    img_scale_w = [s[1] for s in self.img_scales]
                    new_h = np.random.randint(
                        min(img_scale_h),
                        max(img_scale_h) + 1
                    )
                    new_w = np.random.randint(
                        min(img_scale_w),
                        max(img_scale_w) + 1
                    )
            else:
                img_scale = self.img_scales[np.random.randint(self.num_scales)]
                if self.keep_ratio:
                    scale_factor = min(min(img_scale) / min(h, w),
                                       max(img_scale) / max(h, w))
                    new_h = int(h * float(scale_factor) + 0.5)
                    new_w = int(w * float(scale_factor) + 0.5)
                else:
                    new_h = img_scale[0]
                    new_w = img_scale[1]
        else:
            assert self.sample_style == "choice"
            img_scale = self.img_scales[np.random.randint(self.num_scales)]
            if self.keep_ratio:
                scale_factor = min(min(img_scale) / min(h, w),
                                   max(img_scale) / max(h, w))
                new_h = int(h * float(scale_factor) + 0.5)
                new_w = int(w * float(scale_factor) + 0.5)
            else:
                new_h = img_scale[0]
                new_w = img_scale[1]

        return ResizeTransformVideo(
            imgs[0].shape[0], imgs[0].shape[1], new_h, new_w, self.interp
        )


@TRANSFORMS.register()
class ResizeLongestEdge(TransformGen):
    """
    Scale the longer edge to the given size.
    """

    def __init__(self, long_edge_length, sample_style="range", interp=Image.BILINEAR,
                 jitter=(0.0, 32)):
        """
        Args:
            long_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            sample_style (str): either "range" or "choice".
            interp: PIL interpolation method.
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(long_edge_length, int):
            long_edge_length = (long_edge_length, long_edge_length)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        if self.is_range:
            size = np.random.randint(
                self.long_edge_length[0], self.long_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.long_edge_length)
        if size == 0:
            return NoOpTransform()

        if self.jitter[0] > 0:
            dw = self.jitter[0] * w
            dh = self.jitter[0] * h
            size = max(h, w) + np.random.uniform(low=-max(dw, dh), high=max(dw, dh))
            size -= size % self.jitter[1]

        scale = size * 1.0 / max(h, w)
        if h < w:
            newh, neww = scale * h, size
        else:
            newh, neww = size, scale * w

        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        return ResizeTransform(h, w, newh, neww, self.interp)


@TRANSFORMS.register()
class ResizeShortestEdge(TransformGen):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self,
        short_edge_length,
        max_size=sys.maxsize,
        sample_style="range",
        interp=Image.BILINEAR,
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
            interp: PIL interpolation method.
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]

        if self.is_range:
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


@TRANSFORMS.register()
class EfficientDetResizeCrop(TransformGen):
    """
       Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
       If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
       """
    def __init__(
            self, size, scale, interp=Image.BILINEAR
    ):
        """
        Args:
        """
        super().__init__()
        self.target_size = (size, size)
        self.scale = scale
        self.interp = interp

    def get_transform(self, img, annotations=None):
        # Select a random scale factor.
        scale_factor = np.random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]
        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.shape[1], img.shape[0]
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * np.random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * np.random.uniform(0, 1))
        return EfficientDetResizeCropTransform(
            scaled_h, scaled_w, offset_y, offset_x, img_scale, self.target_size, self.interp)


@TRANSFORMS.register()
class RandomCrop(TransformGen):
    """
    Randomly crop a subimage out of an image.
    """

    def __init__(self, crop_type: str, crop_size, strict_mode=True):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
                - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
                  size (H, W). crop size should be in (0, 1]
                - "relative_range": uniformly sample two values from [crop_size[0], 1]
                  and [crop_size[1]], 1], and use them as in "relative" crop type.
                - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
                  crop_size must be smaller than the input image size.
                - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
                  [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
                  Then crop a region (H_crop, W_crop).
            crop_size (tuple[float]): the relative ratio or absolute pixels of
                height and width
            strict_mode (bool): if `True`, the target `crop_size` must be smaller than
                the original image size.
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        if self.strict_mode:
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
                self
            )
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


@TRANSFORMS.register()
class RandomCropWithInstance(RandomCrop):
    """
    Make sure the cropping region contains the center of a random instance from annotations.
    """

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        if self.strict_mode:
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
                self
            )
        offset_range_h = max(h - croph, 0)
        offset_range_w = max(w - cropw, 0)
        # Make sure there is always at least one instance in the image
        assert annotations is not None, "Can not get annotations infos."
        instance = np.random.choice(annotations)
        bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
        bbox = torch.tensor(bbox)
        center_xy = (bbox[:2] + bbox[2:]) / 2

        offset_range_h_min = max(center_xy[1] - croph, 0)
        offset_range_w_min = max(center_xy[0] - cropw, 0)
        offset_range_h_max = min(offset_range_h, center_xy[1] - 1)
        offset_range_w_max = min(offset_range_w, center_xy[0] - 1)

        h0 = np.random.randint(offset_range_h_min, offset_range_h_max + 1)
        w0 = np.random.randint(offset_range_w_min, offset_range_w_max + 1)
        return CropTransform(w0, h0, cropw, croph)


@TRANSFORMS.register()
class RandomCropWithMaxAreaLimit(RandomCrop):
    """
    Find a cropping window such that no single category occupies more than
    `single_category_max_area` in `sem_seg`.

    The function retries random cropping 10 times max.
    """

    def __init__(self, crop_type: str, crop_size, strict_mode=True,
                 single_category_max_area=1.0, ignore_value=255):
        super().__init__(crop_type, crop_size, strict_mode)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        if self.single_category_max_area >= 1.0:
            crop_tfm = super().get_transform(img, annotations)
        else:
            h, w = img.shape[:2]
            assert "sem_seg" in annotations[0]
            sem_seg = annotations[0]["sem_seg"]
            croph, cropw = self.get_crop_size((h, w))
            for _ in range(10):
                y0 = np.random.randint(h - croph + 1)
                x0 = np.random.randint(w - cropw + 1)
                sem_seg_temp = sem_seg[y0: y0 + croph, x0: x0 + cropw]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_value]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.single_category_max_area:
                    break
            crop_tfm = CropTransform(x0, y0, cropw, croph)
        return crop_tfm


@TRANSFORMS.register()
class RandomCropPad(RandomCrop):
    """
    Randomly crop and pad a subimage out of an image.
    """
    def __init__(self,
                 crop_type: str,
                 crop_size,
                 img_value=None,
                 seg_value=None):
        super().__init__(crop_type, crop_size, strict_mode=False)
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        h0 = np.random.randint(h - croph + 1) if h >= croph else 0
        w0 = np.random.randint(w - cropw + 1) if w >= cropw else 0
        dh = min(h, croph)
        dw = min(w, cropw)
        # print(w0, h0, dw, dh)
        return CropPadTransform(w0, h0, dw, dh, cropw, croph, self.img_value,
                                self.seg_value)


@TRANSFORMS.register()
class RandomExtent(TransformGen):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            scale_range (l, h): Range of input-to-output size scaling factor.
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        img_h, img_w = img.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(
                int(src_rect[3] - src_rect[1]),
                int(src_rect[2] - src_rect[0]),
            ),
        )


@TRANSFORMS.register()
class RandomContrast(TransformGen):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation.
            intensity_max (float): Maximum augmentation.
            prob (float): probability of transforms image contrast.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        do = self._rand_range() < self.prob
        if do:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            return BlendTransform(src_image=img.mean(), src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class RandomBrightness(TransformGen):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.):
        """
        Args:
            intensity_min (float): Minimum augmentation.
            intensity_max (float): Maximum augmentation.
            prob (float): probability of transforms image brightness.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        do = self._rand_range() < self.prob
        if do:
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class RandomSaturation(TransformGen):
    """
    Randomly transforms image saturation.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max, prob=1.0):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
            prob (float): probability of transforms image saturation.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        do = self._rand_range() < self.prob
        if do:
            assert img.shape[-1] == 3, "Saturation only works on RGB images"
            w = np.random.uniform(self.intensity_min, self.intensity_max)
            grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
            return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class RandomLighting(TransformGen):
    """
    Randomly transforms image color using fixed PCA over ImageNet.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, img, annotations=None):
        assert img.shape[-1] == 3, "Saturation only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals),
            src_weight=1.0,
            dst_weight=1.0,
        )


@TRANSFORMS.register()
class RandomSwapChannels(TransformGen):
    """
    Randomly swap image channels.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): probability of swap channels.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        _, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return RandomSwapChannelsTransform()
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class MinIoURandomCrop(TransformGen):
    """
    Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        """
        Args:
            min_ious (tuple): minimum IoU threshold for all intersections with bounding boxes
            min_crop_size (float): minimum crop's size
                (i.e. h,w := a*h, a*w, where a >= min_crop_size).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations):
        """
        Args:
            img (ndarray): of shape HxWxC(RGB). The array can be of type uint8
                in range [0, 255], or floating point in range [0, 255].
            annotations (list[dict[str->str]]):
                Each item in the list is a bbox label of an object. The object is
                    represented by a dict,
                which contains:
                 - bbox (list): bbox coordinates, top left and bottom right.
                 - bbox_mode (str): bbox label mode, for example: `XYXY_ABS`,
                    `XYWH_ABS` and so on...
        """
        sample_mode = (1, *self.min_ious, 0)
        h, w = img.shape[:2]

        boxes = list()
        for obj in annotations:
            boxes.append(BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS))
        boxes = torch.tensor(boxes)

        while True:
            mode = np.random.choice(sample_mode)
            if mode == 1:
                return NoOpTransform()

            min_iou = mode
            for i in range(50):
                new_w = np.random.uniform(self.min_crop_size * w, w)
                new_h = np.random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = np.random.uniform(w - new_w)
                top = np.random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))

                overlaps = pairwise_iou(
                    Boxes(patch.reshape(-1, 4)),
                    Boxes(boxes.reshape(-1, 4))
                )

                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1])
                        * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                return IoUCropTransform(int(left), int(top), int(new_w), int(new_h))


@TRANSFORMS.register()
class Expand(TransformGen):
    """
    Random Expand the image & bboxes.
    """

    def __init__(self, ratio_range=(1, 4), mean=(0, 0, 0), prob=0.5):
        """
        Args:
            ratio_range (tuple): range of expand ratio.
            mean (tuple): mean value of datasets.
            prob (float): probability of applying this transformation.
        """
        super().__init__()
        self._init(locals())
        self.min_ratio, self.max_ratio = ratio_range

    def get_transform(self, img, annotations=None):
        if np.random.uniform(0, 1) > self.prob:
            return NoOpTransform()
        h, w, c = img.shape
        ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))
        return ExpandTransform(left, top, ratio, self.mean)


@TRANSFORMS.register()
class Lightning(TransformGen):
    """
    Lightning augmentation, used in classification.

    .. code-block:: python
        tfm = LightningTransform(
            alpha_std=0.1,
            eig_val=np.array([[0.2175, 0.0188, 0.0045]]),
            eig_vec=np.array([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203]
            ]),
        )
    """
    def __init__(self, alpha_std, eig_val, eig_vec):
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return LightningTransform(self.alpha_std, self.eig_val, self.eig_vec)


@TRANSFORMS.register()
class RandomScale(TransformGen):
    """
    Randomly scale the image according to the specified output size and scale ratio range.

    This transform has the following three steps:

        1. select a random scale factor according to the specified scale ratio range.
        2. recompute the accurate scale_factor using rounded scaled image size.
        3. select non-zero random offset (x, y) if scaled image is larger than output_size.
    """

    def __init__(self, output_size, ratio_range=(0.1, 2), interp="BILINEAR"):
        """
        Args:
            output_size (tuple): image output size.
            ratio_range (tuple): range of scale ratio.
            interp (str): the interpolation method. Options includes:
              * "NEAREST"
              * "BILINEAR"
              * "BICUBIC"
              * "LANCZOS"
              * "HAMMING"
              * "BOX"
        """
        super().__init__()
        self._init(locals())
        self.min_ratio, self.max_ratio = ratio_range
        if isinstance(self.output_size, int):
            self.output_size = [self.output_size] * 2

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        output_h, output_w = self.output_size

        # 1. Select a random scale factor.
        random_scale_factor = np.random.uniform(self.min_ratio, self.max_ratio)

        scaled_size_h = int(random_scale_factor * output_h)
        scaled_size_w = int(random_scale_factor * output_w)

        # 2. Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_h = scaled_size_h * 1.0 / h
        image_scale_w = scaled_size_w * 1.0 / w
        image_scale = min(image_scale_h, image_scale_w)

        # 3. Select non-zero random offset (x, y) if scaled image is larger than output_size.
        scaled_h = int(h * 1.0 * image_scale)
        scaled_w = int(w * 1.0 * image_scale)

        return ScaleTransform(h, w, scaled_h, scaled_w, self.interp)


@TRANSFORMS.register()
class Pad(TransformGen):
    """
    Pad image with `pad_value` to the specified `target_h` and `target_w`.

    Adds `top` rows of `pad_value` on top, `left` columns of `pad_value` on the left,
    and then pads the image on the bottom and right with `pad_value` until it has
    dimensions `target_h`, `target_w`.

    This op does nothing if `top` and `left` is zero and the image already has size
    `target_h` by `target_w`.
    """

    def __init__(self, top, left, target_h, target_w, pad_value=0):
        """
        Args:
            top (int): number of rows of `pad_value` to add on top.
            left (int): number of columns of `pad_value` to add on the left.
            target_h (int): height of output image.
            target_w (int): width of output image.
            pad_value (int): the value used to pad the image.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return PadTransform(self.top, self.left, self.target_h, self.target_w, self.pad_value)


@TRANSFORMS.register()
class RandomList(TransformGen):
    """
    Random select subset of provided augmentations.
    """
    def __init__(self, transforms, num_layers=2, choice_weights=None):
        """
        Args:
            transforms (List[TorchTransformGen]): list of transforms need to be performed.
            num_layers (int): parameters of np.random.choice.
            choice_weights (optional, float): parameters of np.random.choice.
        """
        self.transforms = transforms
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def get_transform(self, img, annotations=None):
        tfms = np.random.choice(
            self.transforms,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights)
        return ComposeTransform(tfms)


@TRANSFORMS.register()
class ShuffleList(TransformGen):
    """
    Randomly shuffle the `transforms` order.
    """

    def __init__(self, transforms):
        """
        Args:
            transforms (list[TransformGen]): List of transform to be shuffled.
        """
        super().__init__()
        self.transforms = transforms

    def get_transform(self, img, annotations=None):
        np.random.shuffle(self.transforms)
        return ComposeTransform(self.transforms)


@TRANSFORMS.register()
class RepeatList(TransformGen):
    """
    Forward several times of provided transforms for a given image.
    """
    def __init__(self, transforms, repeat_times):
        """
        Args:
            transforms (list[TransformGen]): List of transform to be repeated.
            repeat_times (int): number of duplicates desired.
        """
        super().__init__()
        self.transforms = transforms
        self.times = repeat_times

    def get_transform(self, img, annotations=None):
        return ComposeTransform(self.transforms)

    def __call__(self, img, annotations=None):
        repeat_imgs = []
        repeat_annotations = []
        for t in range(self.times):
            tmp_img, tmp_anno = self.get_transform(img)(img, annotations)
            repeat_imgs.append(tmp_img)
            repeat_annotations.append(tmp_anno)
        repeat_imgs = np.stack(repeat_imgs, axis=0)
        return repeat_imgs, repeat_annotations


@TRANSFORMS.register()
class JigsawCrop(TransformGen):
    """
    Crop an image into n_grid times n_grid crops, no overlaps.
    """
    def __init__(self, n_grid=3, img_size=255, crop_size=64):
        """
        Args:
            n_grid (int): crop image into n_grid x n_grid crops.
            img_size (int): image scale size.
            crop_size (int): crop size.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return JigsawCropTransform(self.n_grid, self.img_size, self.crop_size)

    def __call__(self, img, annotations=None):
        crops, annos = self.get_transform(img)(img, annotations)
        return np.stack(crops, axis=0), annos


@TRANSFORMS.register()
class NoneOpTransform(TransformGen):
    def __init__(self,):
        super(NoneOpTransform, self).__init__()

    def get_transform(self, img, annotations=None):
        return NoOpTransform()


@TRANSFORMS.register()
class ComposeListWithProp(TransformGen):
    """
    Random select subset of provided augmentations.
    """
    def __init__(self, transforms, num_layers=2, choice_weights=None):
        """
        Args:
            transforms (List[TorchTransformGen]): list of transforms need to be performed.
            num_layers (int): parameters of np.random.choice.
            choice_weights (optional, float): parameters of np.random.choice.
        """
        self.transforms = transforms
        self.num_layers = num_layers
        assert len(self.transforms) == self.num_layers
        self.choice_weights = choice_weights

    def get_transform(self, img, annotations=None):
        if np.random.rand() > 0.5:
            tfms = ComposeTransform(self.transforms)
        else:
            tfms = NoneOpTransform()

        return tfms

@TRANSFORMS.register()
class DETRCrop(TransformGen):
    """
    DETR format transform gen including random resize and random crop.
    """
    def __init__(
            self,
            crop_type,
            crop_size,
            short_edge_length,
            max_size=sys.maxsize,
            sample_style="range",
            interp=Image.BILINEAR):
        super(DETRCrop, self).__init__()
        self.resize = ResizeShortestEdge(short_edge_length,
                                         max_size,
                                         sample_style,
                                         interp)
        self.crop = RandomCrop(crop_type,
                               crop_size)

    def get_transform(self, img, annotations=None):
        pass

    def __call__(self, img, annotations=None):
        if np.random.rand() > 0.5:
            img, annotations = self.resize.get_transform(img, annotations)(img, annotations)
            return self.crop.get_transform(img, annotations)(img, annotations)
        else:
            return img, annotations
