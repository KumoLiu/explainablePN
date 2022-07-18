import math
import numpy as np
from datasets import CLASSIFICATION_DATASETS
from scipy.ndimage import generate_binary_structure, binary_erosion
from strix.data_io.base_dataset.classification_dataset import BasicClassificationDataset
from strix.configures import config as cfg


from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *

"""
MP/S 课题数据集
MP指的是微乳头型肺腺癌，S指的是实体型肺腺癌
"""

@CLASSIFICATION_DATASETS.register('2D', 'jsph_mps_paper',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_mps_dataset(files_list, phase, opts, (84,84))

@CLASSIFICATION_DATASETS.register('2D', 'jsph_mps_smaller_paper',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new.json")
def get_25d_dataset(files_list, phase, opts):
    return get_lung_mps_dataset(files_list, phase, opts, (64,64))

@CLASSIFICATION_DATASETS.register('3D', 'jsph_mps_smaller_paper',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new_addprob.json")
def get_25d_dataset(files_list, phase, opts):
    return get_3d_dataset(files_list, phase, opts, (64, 64, 32), use_prob=False, divide_mask=True)

#! final version
@CLASSIFICATION_DATASETS.register('3D', 'jsph_mps_smaller_prob_paper',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new_addprob.json")
def get_25d_dataset(files_list, phase, opts):
    return get_3d_dataset(files_list, phase, opts, (64, 64, 32), use_prob=True, divide_mask=False)

@CLASSIFICATION_DATASETS.register('3D', 'jsph_mps_smaller_seg_paper',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new_addprob.json")
def get_25d_dataset(files_list, phase, opts):
    return get_3d_dataset(files_list, phase, opts, (64, 64, 32), use_prob=False, use_seg=True, divide_mask=False)


@CLASSIFICATION_DATASETS.register('2D', 'jsph_mps_larger',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new.json")
def get_25d_dataset_larger(files_list, phase, opts):
    return get_lung_mps_dataset(files_list, phase, opts, (96,96), use_prob=False, divide_mask=True)


@CLASSIFICATION_DATASETS.register('3D', 'jsph_mps_3d_smaller',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new.json")
def get_3d_dataset_smaller(files_list, phase, opts):
    return get_3d_dataset(files_list, phase, opts, (64, 64, 32))

# pos weight 0.87, 2.75, 1.85, 2.4
# weight 0.8, 0.85, 0.85, 0.75
@CLASSIFICATION_DATASETS.register('3D', 'jsph_mps_3d',
    "/homes/yliu/Data/pn_cls_data/MPS/crop/datalist-train-mps_new.json")
def get_3d_dataset_larger(files_list, phase, opts):
    return get_3d_dataset(files_list, phase, opts, (96, 96, 32))


def get_3d_dataset(files_list, phase, opts, spatial_size=(96, 96, 32), use_prob=False, use_seg=False, divide_mask=False):
    spacing = opts.get('spacing', (0.7, 0.7, 1.5))
    preload = opts.get('preload', 0)
    augment_ratio = opts.get('augment_ratio', 0.4)
    image_keys = opts.get('image_keys', ['image'])
    mask_keys = opts.get('mask_keys', ['mask'])
    feat_keys = opts.get('feat_keys', ['features'])
    if use_prob:
        mask_keys = ['probmap'] 
    elif use_seg:
        mask_keys = ['seg']
    else: 
        mask_keys = ['mask']
    cfg.set_key('MASK', mask_keys[0])

    use_mask = False

    if phase == 'train':
        additional_transforms = [
            RandRotate90D(
                keys=image_keys+mask_keys,
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
        CenterSpatialCropD(
            keys=image_keys+mask_keys,
            roi_size=spatial_size
        )
    ]
    if divide_mask:
        cropper += [DivideMaskD(mask_key=mask_keys[0], image_key=image_keys[0], threshold=0.23, dim=3)]

    
    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadImageD(keys=image_keys+mask_keys, dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+mask_keys),
        orienter=None,  # Orientationd(keys=['image','mask'], axcodes=orientation),
        spacer=None,
        rescaler=None,
        resizer=None,
        cropper=cropper,
        caster=CastToTyped(keys=image_keys, dtype=np.float32),
        to_tensor=ToTensorExD(keys=image_keys+mask_keys+feat_keys),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
        additional_transforms=None,
    )

    return dataset

class DivideMaskD(MapTransform):
    """
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
    """
    def __init__(
        self, mask_key, image_key, dim=2, lung_key=None, threshold=None, scale=2.5
    ) -> None:
        super().__init__(mask_key)
        self.dim = dim
        self.image_key = image_key
        self.threshold = threshold
        self.lung_key = lung_key
        self.scale = scale

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.threshold is None:
                mean_, std_ = np.mean(d[self.image_key][d[key]>0]), np.std(d[self.image_key][d[key]>0])
                foreground = np.logical_and(
                    max(0.2, mean_-self.scale*std_)<=d[self.image_key], d[self.image_key]<=mean_+2*self.scale*std_
                ).astype(np.int8)
            else:
                foreground = (d[self.image_key]>self.threshold).astype(np.int8)
            
            if self.lung_key is not None:
                lung_mask = binary_erosion(
                    d[self.lung_key].squeeze(), structure=generate_binary_structure(3,1), iterations=2
                )[np.newaxis,...]
                foreground[np.logical_not(lung_mask)] = 0
            
            foreground[d[key]>0] = 2
            if self.dim == 2:
                new_foreground = foreground[foreground.shape[0]//2,...][np.newaxis,...]
                new_foreground = np.eye(3)[new_foreground].transpose(3,0,1,2)
                d[key] = new_foreground.squeeze()
            else:
                d[key] = np.eye(3)[foreground.squeeze(0)].transpose(3,0,1,2)
        return d


def get_lung_mps_dataset(files_list, phase, opts, spatial_size):
    # median reso: 0.70703125 z_reso: 1.5
    # 0.5 & 99.5 percentile: -1024, 388
    spacing = opts.get('spacing', (0.7, 0.7, 1.5))
    in_channels = opts.get('input_nc', 3)
    preload = opts.get('preload', 0)
    augment_ratio = opts.get('augment_ratio', 0.5)
    orientation = opts.get('orientation', 'RAI')
    image_keys = opts.get('image_keys', ['image'])
    mask_keys = opts.get('mask_keys', ['mask'])
    feat_keys = opts.get('feat_keys', ['features'])

    use_mask = False
    divide_mask = False
    crop_mode = 'parallel'
    center_mode = 'maximum'


    if phase == 'train':
        additional_transforms = [
            RandRotate90D(
                keys=image_keys+mask_keys,
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
            center_mode='maximum',
            z_axis=2,
            n_slices=in_channels,
        ),
    ]

    if divide_mask:
        cropper += [
                DivideMaskD(mask_key=mask_keys[0], image_key=image_keys[0], threshold=0.25, dim=2),
            ]
    else:
        rescaler = None


    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadImageD(keys=image_keys+mask_keys, dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+mask_keys),
        orienter=None,
        spacer=None,
        rescaler=None,
        resizer=None,
        cropper=cropper,
        caster=CastToTyped(keys=image_keys, dtype=np.float32),
        to_tensor=ToTensorExD(keys=image_keys+mask_keys+feat_keys),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
        additional_transforms=additional_transforms,
    )
    
    return dataset
