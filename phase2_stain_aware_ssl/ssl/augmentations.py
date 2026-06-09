#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Augmentation pipelines for DINO-style SSL on PNG patch datasets."""

from __future__ import annotations

import random
from typing import Any, Dict, List

import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    def __init__(self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.p = float(p)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class Solarization:
    def __init__(self, p: float = 0.0, threshold: int = 128) -> None:
        self.p = float(p)
        self.threshold = int(threshold)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return ImageOps.solarize(img, threshold=self.threshold)


def _crop_pipeline(
    crop_size: int,
    scale: tuple[float, float],
    *,
    hflip_p: float,
    vflip_p: float,
    color_jitter_p: float,
    jitter_b: float,
    jitter_c: float,
    jitter_s: float,
    jitter_h: float,
    grayscale_p: float,
    blur_p: float,
    blur_radius_min: float,
    blur_radius_max: float,
    solarize_p: float,
    solarize_threshold: int,
) -> T.Compose:
    color_jitter = T.ColorJitter(jitter_b, jitter_c, jitter_s, jitter_h)
    return T.Compose(
        [
            T.RandomResizedCrop(size=crop_size, scale=scale, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=hflip_p),
            T.RandomVerticalFlip(p=vflip_p),
            T.RandomApply([color_jitter], p=color_jitter_p),
            T.RandomGrayscale(p=grayscale_p),
            GaussianBlur(p=blur_p, radius_min=blur_radius_min, radius_max=blur_radius_max),
            Solarization(p=solarize_p, threshold=solarize_threshold),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MultiCropTransform:
    """DINO-style transform returning multiple crops from one image."""

    def __init__(self, aug_cfg: Dict[str, Any]) -> None:
        self.n_global = int(aug_cfg.get("N_GLOBAL_CROPS", 2))
        self.n_local = int(aug_cfg.get("N_LOCAL_CROPS", 6))
        global_size = int(aug_cfg.get("GLOBAL_CROP_SIZE", 224))
        local_size = int(aug_cfg.get("LOCAL_CROP_SIZE", 96))
        global_scale = tuple(aug_cfg.get("GLOBAL_CROPS_SCALE", [0.32, 1.0]))
        local_scale = tuple(aug_cfg.get("LOCAL_CROPS_SCALE", [0.05, 0.32]))

        hflip_p = float(aug_cfg.get("H_FLIP_P", 0.5))
        vflip_p = float(aug_cfg.get("V_FLIP_P", 0.5))
        jitter_p = float(aug_cfg.get("COLOR_JITTER_P", 0.8))
        jitter_b = float(aug_cfg.get("COLOR_JITTER_BRIGHTNESS", 0.3))
        jitter_c = float(aug_cfg.get("COLOR_JITTER_CONTRAST", 0.3))
        jitter_s = float(aug_cfg.get("COLOR_JITTER_SATURATION", 0.2))
        jitter_h = float(aug_cfg.get("COLOR_JITTER_HUE", 0.05))
        grayscale_p = float(aug_cfg.get("GRAYSCALE_P", 0.2))

        blur_min = float(aug_cfg.get("BLUR_RADIUS_MIN", 0.1))
        blur_max = float(aug_cfg.get("BLUR_RADIUS_MAX", 2.0))
        blur_g1 = float(aug_cfg.get("BLUR_GLOBAL1_P", 1.0))
        blur_g2 = float(aug_cfg.get("BLUR_GLOBAL2_P", 0.1))
        blur_l = float(aug_cfg.get("BLUR_LOCAL_P", 0.5))
        solarize_p = float(aug_cfg.get("SOLARIZE_P", 0.2))
        solarize_threshold = int(aug_cfg.get("SOLARIZE_THRESHOLD", 128))

        self.global1 = _crop_pipeline(
            global_size,
            global_scale,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
            color_jitter_p=jitter_p,
            jitter_b=jitter_b,
            jitter_c=jitter_c,
            jitter_s=jitter_s,
            jitter_h=jitter_h,
            grayscale_p=grayscale_p,
            blur_p=blur_g1,
            blur_radius_min=blur_min,
            blur_radius_max=blur_max,
            solarize_p=0.0,
            solarize_threshold=solarize_threshold,
        )
        self.global2 = _crop_pipeline(
            global_size,
            global_scale,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
            color_jitter_p=jitter_p,
            jitter_b=jitter_b,
            jitter_c=jitter_c,
            jitter_s=jitter_s,
            jitter_h=jitter_h,
            grayscale_p=grayscale_p,
            blur_p=blur_g2,
            blur_radius_min=blur_min,
            blur_radius_max=blur_max,
            solarize_p=solarize_p,
            solarize_threshold=solarize_threshold,
        )
        self.local = _crop_pipeline(
            local_size,
            local_scale,
            hflip_p=hflip_p,
            vflip_p=vflip_p,
            color_jitter_p=jitter_p,
            jitter_b=jitter_b,
            jitter_c=jitter_c,
            jitter_s=jitter_s,
            jitter_h=jitter_h,
            grayscale_p=grayscale_p,
            blur_p=blur_l,
            blur_radius_min=blur_min,
            blur_radius_max=blur_max,
            solarize_p=0.0,
            solarize_threshold=solarize_threshold,
        )

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        views: List[torch.Tensor] = []
        if self.n_global >= 1:
            views.append(self.global1(img))
        if self.n_global >= 2:
            views.append(self.global2(img))
        for _ in range(max(0, self.n_global - 2)):
            views.append(self.global2(img))
        for _ in range(self.n_local):
            views.append(self.local(img))
        return views


class TwoViewEvalTransform:
    """Deterministic two-view transform for validation DINO loss."""

    def __init__(self, image_size: int = 224) -> None:
        self.view1 = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.view2 = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        return [self.view1(img), self.view2(img)]


class SingleViewEvalTransform:
    """Deterministic single-view transform for embedding extraction."""

    def __init__(self, image_size: int = 224) -> None:
        self.t = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.t(img)


def build_train_transform(cfg: Dict[str, Any]) -> MultiCropTransform:
    return MultiCropTransform(cfg["AUG"])


def build_val_transform(cfg: Dict[str, Any]) -> TwoViewEvalTransform:
    image_size = int(cfg["MODEL"]["BACKBONE"].get("IMAGE_SIZE", 224))
    return TwoViewEvalTransform(image_size=image_size)


def build_eval_transform(cfg: Dict[str, Any]) -> SingleViewEvalTransform:
    image_size = int(cfg["MODEL"]["BACKBONE"].get("IMAGE_SIZE", 224))
    return SingleViewEvalTransform(image_size=image_size)

