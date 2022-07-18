import os, math
import numpy as np

from datasets import CLASSIFICATION_DATASETS
from medlp.data_io.base_dataset.classification_dataset import BasicClassificationDataset
from medlp.configures import config as cfg

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *
from utils import binarization
from datasets.jsph_mps_datasets import DivideMaskD

# import monai_ex
# monai_ex.utils.set_determinism(seed=42, additional_settings=None)

@CLASSIFICATION_DATASETS.register('2D', 'lidc-82-N',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/train_datalist_8-2_cls-mean-RM1.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset(files_list, phase, opts, (32,32), False)


@CLASSIFICATION_DATASETS.register('2D', 'lidc-82-N-two-head-radiomics',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/train_datalist_8-2_cls-mean-RM1-v2-radiomics.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset(files_list, phase, opts, (32,32), False)

@CLASSIFICATION_DATASETS.register('2D', 'lidc-82-N-new',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/RM1-featworound/train_datalist_8-2_cls-new-radiomics.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset(files_list, phase, opts, (32,32), False)


@CLASSIFICATION_DATASETS.register('2D', 'lidc-82',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/non-normed/train_datalist_8-2_cls-mean-RM1.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset(files_list, phase, opts, (32,32), False)

@CLASSIFICATION_DATASETS.register('2D', 'lidc-paper',
    "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-minmax/train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3(ver2)_feature_sphericity.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3_feature_sphericity.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset(files_list, phase, opts, (32,32), False)

#! Subtlety Calcification Texture Margin Sphericity label
#! train 245/736 196/785 74/907 157/824 541/440 791/190
#! test 56/189 54/191 18/227 31/214 137/108 193/52
#! pos weight 0.33, 0.25, 0.08, 0.2, 1.23
@CLASSIFICATION_DATASETS.register('3D', 'lidc-paper',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_0.2_cls_int.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3(ver2)_feature_sphericity.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3_feature_sphericity.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset_3D(files_list, phase, opts, (64, 64, 32), False)

@CLASSIFICATION_DATASETS.register('3D', 'lidc-paper_new-sphericity',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity_addprob.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset_3D(files_list, phase, opts, (64, 64, 32), use_mask=False, use_prob=False, use_seg=False, divide_mask=False)

#! final version
@CLASSIFICATION_DATASETS.register('3D', 'lidc-paper_new-sphericity_prob',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity_addprob.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset_3D(files_list, phase, opts, (64, 64, 32), use_mask=False, use_prob=True, use_seg=False, divide_mask=False)

@CLASSIFICATION_DATASETS.register('3D', 'lidc-paper_new-sphericity_seg',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity_addprob.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset_3D(files_list, phase, opts, (64, 64, 32), use_mask=False, use_prob=False, use_seg=True, divide_mask=False)
# def get_25d_dataset(files_list, phase, opts):
#     return get_lung_dataset_3D(files_list, phase, opts, (64, 64, 32), use_mask=False, use_prob=False, use_seg=True, divide_mask=True)


@CLASSIFICATION_DATASETS.register('3D', 'lidc-paper-prob',
    # "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_0.2_cls_int_addprob.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-minmax/train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_7-3_minmax_remove3(ver3)_feature_sphericity_addprob.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset_3D(files_list, phase, opts, (64, 64, 32), False)

@CLASSIFICATION_DATASETS.register('3D', 'lidc-paper-prob-smaller',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_0.2_cls_int_addprob.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3(ver2)_feature_sphericity.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3_feature_sphericity.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset_3D(files_list, phase, opts, (32, 32, 16), False)

@CLASSIFICATION_DATASETS.register('3D', 'lidc-paper-smaller',
    "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/train_datalist_0.2_cls_int.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3(ver2)_feature_sphericity.json")
    # "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data-zscore/train_datalist_8-2_minmax_remove3_feature_sphericity.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_dataset_3D(files_list, phase, opts, (48, 48, 16), False)




def get_lung_dataset(files_list, phase, opts, spatial_size, use_mask):
    # median reso: 0.70703125 z_reso: 1.5
    # 0.5 & 99.5 percentile: -1024, 388
    spacing = opts.get('spacing', (0.7, 0.7, 1))
    in_channels = opts.get('input_nc', 3)
    preload = opts.get('preload', 0)
    augment_ratio = opts.get('augment_ratio', 0.4)
    image_keys = opts.get('image_keys', ['image'])
    mask_keys = opts.get('mask_keys', ['mask'])

    use_feat = True  #True
    use_mask = use_mask
    crop_mode = 'parallel'
    center_mode = 'maximum'

    if use_feat:
        feat_keys = opts.get('feat_keys', ['features'])

    if phase == 'train':
        additional_transforms = [
            RandRotate90D(
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
            crop_mode='parallel',
            center_mode=center_mode,
            z_axis=2,
            n_slices=in_channels,
        ),

        # GetMaxSlices3direcCropD(
        #     keys=image_keys+mask_keys, 
        #     mask_key=mask_keys[0],
        #     roi_size=spatial_size, 
        #     crop_mode=crop_mode, 
        #     center_mode='center',
        #     n_slices=in_channels),
    ]

    cache_dir_name = 'cache' if not use_mask else 'cache-mask'
    cache_dir_name += f'-{crop_mode}-{center_mode}-{spatial_size}'
    # cache_dir = check_dir(os.path.dirname(opts.get('experiment_path')), cache_dir_name)

    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadImageD(keys=image_keys+mask_keys, dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+mask_keys),
        orienter=None,  # Orientationd(keys=['image','mask'], axcodes=orientation),
        spacer=SpacingD(keys=image_keys+mask_keys, pixdim=spacing, mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
        rescaler=[
            # NormalizeIntensityD(keys=image_keys),
            LambdaD(keys='label', func=lambda x: int(x>3)),
        ],
        resizer=None,
        cropper=cropper,
        caster=CastToTyped(keys=image_keys, dtype=np.float32),
        to_tensor=ToTensorExD(keys=image_keys+mask_keys+feat_keys) if use_feat else ToTensorExD(keys=image_keys+mask_keys),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
        # dataset_type=PersistentDataset,
        # dataset_kwargs={'cache_dir': cache_dir},
        additional_transforms=additional_transforms,
    )

    return dataset

def get_lung_dataset_3D(files_list, phase, opts, spatial_size, use_mask, use_prob, use_seg, divide_mask=False):
    # medean reso: 0.6935771 z_reso: 1.7325449  LIDC-IDRI
    # median reso: 0.6935771 z_reso: 1.25  LIDC-IDRI
    # 0.5 & 99.5 percentile: -1024, 388
    spacing = opts.get('spacing', (0.7, 0.7, 1.25))
    preload = opts.get('preload', 0)
    augment_ratio = opts.get('augment_ratio', 0.4)
    image_keys = opts.get('image_keys', ['image'])
    if use_prob:
        mask_keys = ['probmap'] 
    elif use_seg:
        mask_keys = ['seg']
    else: 
        mask_keys = ['mask']
    cfg.set_key('MASK', mask_keys[0])
    

    use_feat = True  #True
    use_mask = use_mask
    crop_mode = 'parallel'
    center_mode = 'center'

    if use_feat:
        feat_keys = opts.get('feat_keys', ['features'])

    if phase == 'train':
        additional_transforms = [
            RandFlipD(
                keys=image_keys+mask_keys,
                prob=augment_ratio,
                spatial_axis=0
            ),
            RandRotate90D(
                keys=image_keys+mask_keys,
                prob=augment_ratio,
            ),
            RandAdjustContrastD(
                keys=image_keys,
                prob=augment_ratio,
                gamma=(0.9, 1.1)
            ),
            # RandAffineD(
            #     keys=image_keys+mask_keys,
            #     prob=augment_ratio,
            #     rotate_range=[math.pi/30, math.pi/30],
            #     shear_range=[0.1, 0.1],
            #     translate_range=[1, 1],
            #     scale_range=[0.1, 0.1],
            #     mode=["bilinear"],
            #     padding_mode=['reflection'],
            #     as_tensor_output=False
            # )
        ]
        # if not use_mask:
        #     additional_transforms += [
        #         RandGaussianNoiseD(
        #             keys=image_keys,
        #             prob=augment_ratio,
        #             mean=0.0,
        #             std=0.05,
        #         ),
        #     ]
    elif phase in ['valid', 'test']:
        additional_transforms = []
    elif phase == 'test_wo_label':
        raise NotImplementedError


    cropper = [
        CenterSpatialCropd(
            keys=image_keys+mask_keys,
            roi_size=spatial_size,
        ),
        # RandCenterSpatialCropd(
        #     keys=image_keys+mask_keys,
        #     image_key=image_keys[0],
        #     roi_size=spatial_size,
        #     offset=10
        # ),
        ResizeWithPadOrCropD(
            keys=image_keys+mask_keys,
            spatial_size=spatial_size,
        )
    ]
    if divide_mask:
        cropper += [DivideMaskD(mask_key=mask_keys[0], image_key=image_keys[0], threshold=0.13, dim=3)]

    cache_dir_name = 'cache' if not use_mask else 'cache-mask'
    cache_dir_name += f'-{crop_mode}-{center_mode}-{spatial_size}'
    # cache_dir = check_dir(os.path.dirname(opts.get('experiment_path')), cache_dir_name)

    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadImageD(keys=image_keys+mask_keys, dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+mask_keys),
        orienter=None,  # Orientationd(keys=['image','mask'], axcodes=orientation),
        spacer=SpacingD(keys=image_keys+mask_keys, pixdim=spacing, mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
        rescaler=[
            # NormalizeIntensityD(keys=image_keys),
            LambdaD(keys='label', func=lambda x: int(x>3)),
            # LambdaD(keys=mask_keys[0], func=lambda x: x>0)
        ],
        resizer=None,
        cropper=cropper,
        # caster=CastToTyped(keys=image_keys+mask_keys, dtype=[np.float32, np.int]),
        caster=CastToTyped(keys=image_keys, dtype=np.float32),
        to_tensor=ToTensorExD(keys=image_keys+mask_keys+feat_keys) if use_feat else ToTensorExD(keys=image_keys+mask_keys),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
        # dataset_type=PersistentDataset,
        # dataset_kwargs={'cache_dir': cache_dir},
        additional_transforms=additional_transforms,
    )

    return dataset
