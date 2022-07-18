import math
import numpy as np

from datasets import CLASSIFICATION_DATASETS
from strix.data_io.base_dataset.classification_dataset import BasicClassificationDataset
from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *


"""
脉管侵犯MVI 课题数据集
"""

# 使用预处理后的Crop数据（已做归一化）
@CLASSIFICATION_DATASETS.register('2D', 'jsph_mvi-mask',
    "/homes/yliu/Data/pn_cls_data/MVI/datalist-withfeat-train.json")
def get_25d_dataset(files_list, phase, opts):
    return get_mvi_dataset(files_list, phase, opts, (64,64), True)
    
@CLASSIFICATION_DATASETS.register('2D', 'jsph_mvi',
    "/homes/yliu/Data/pn_cls_data/MVI/datalist-withfeat-train.json")
def get_25d_dataset(files_list, phase, opts):
    return get_mvi_dataset(files_list, phase, opts, (64,64), False)

@CLASSIFICATION_DATASETS.register('2D', 'jsph_mvi_pretrain',
    "/homes/clwang/Data/jsph_lung/MVI/data_crops/datalist-train.json")
def get_25d_dataset(files_list, phase, opts):
    return get_mvi_dataset(files_list, phase, opts, (32,32), False)


def get_mvi_dataset(files_list, phase, opts, spatial_size, use_mask):
    preload = opts.get('preload', 0)
    augment_ratio = opts.get('augment_ratio', 0.4)
    in_channels = opts.get('input_nc', 3)
    preload = opts.get('preload', 0)
    orientation = opts.get('orientation', 'RAI')
    image_keys = opts.get('image_keys', ['image'])
    mask_keys = opts.get('mask_keys', ['mask'])
    feat_keys = opts.get('feat_keys', ['features'])

    use_mask = use_mask
    crop_mode = 'parallel'
    center_mode = 'maximum'

    if phase == 'train':
        additional_transforms = [
            RandRotate90ExD(
                keys=image_keys,
                prob=augment_ratio,
            ),
            RandAdjustContrastD(
                keys=image_keys,
                prob=augment_ratio,
                gamma=(0.9, 1.1)
            ),
            RandAffineD(
                keys=image_keys,
                prob=augment_ratio,
                rotate_range=[math.pi/30, math.pi/30],
                shear_range=[0.1, 0.1],
                translate_range=[1, 1],
                scale_range=[0.1, 0.1],
                mode=["bilinear"],
                padding_mode=['reflection'],
                as_tensor_output=False
            )
        ]
        if not use_mask:
            additional_transforms += [
                RandGaussianNoiseD(
                    keys=image_keys,
                    prob=augment_ratio,
                    mean=0.0,
                    std=0.05,
                ),
            ]
    elif phase in ['valid', 'test']:
        additional_transforms = []
    elif phase == 'test_wo_label':
        raise NotImplementedError

    cropper = [
        LabelMorphologyD(
            keys=mask_keys,
            mode='dilation',
            radius=2,
            binary=True
        ),
        MaskIntensityExD(
            keys=image_keys,
            mask_key='mask',
        ),
    ] if use_mask else []

    cropper += [
        CenterMask2DSliceCropD(
            keys=image_keys+mask_keys,
            mask_key=mask_keys[0],
            roi_size=spatial_size,
            crop_mode=crop_mode,
            center_mode=center_mode,
            z_axis=2,
            n_slices=in_channels,
        )
    ]

    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadImageD(keys=image_keys+mask_keys, dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+mask_keys),
        orienter=None,  # Orientationd(keys=['image','mask'], axcodes=orientation),
        spacer=None,
        rescaler=[
            NormalizeIntensityD(keys=image_keys),
        ],
        resizer=None,
        cropper=cropper,
        caster=CastToTypeExD(keys=image_keys, dtype=np.float32),
        to_tensor=[
            ToTensorExD(keys=image_keys+mask_keys+feat_keys),
        ],
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
        # dataset_type=PersistentDataset,
        # dataset_kwargs={'cache_dir': cache_dir},
        additional_transforms=additional_transforms,
    )

    return dataset
