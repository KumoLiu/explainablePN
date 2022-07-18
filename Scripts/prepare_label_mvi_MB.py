#%% Generate data crops
import json
from pathlib import Path
import nibabel as nib
from ..utils import check_dir, get_items_from_file
from tqdm import tqdm
from medlp.data_io.base_dataset.classification_dataset import BasicClassificationDataset
from monai.data import CacheDataset, PersistentDataset, Dataset
from monai_ex.transforms import *

phases = ['train', 'test']
root_dir = Path('/homes/yliu/Data/pn_cls_data/SHIXINGJIEJIE/MVI_MB.json')
out_dir = Path('/homes/yliu/Data/pn_cls_data/SHIXINGJIEJIE/Crop-Norm-New')
image_keys = ['image']
mask_keys = ['mask']
spacing = (0.7, 0.7, 1)
# spacing = (0.7, 0.7, 1.5)
# 0.5 & 99.5 percentile: 11.0 1387.0

files_list = get_items_from_file(root_dir, format='json')


dataset = BasicClassificationDataset(
    files_list,
    loader=LoadImageD(keys=image_keys+mask_keys, dtype=np.float32),
    channeler=AddChannelD(keys=image_keys+mask_keys),
    orienter=None, 
    # spacer=SpacingD(
    #     keys=image_keys+mask_keys,
    #     pixdim=spacing,
    #     mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]
    # ),
    spacer=None,
    # rescaler=ScaleIntensityRangePercentilesD(
    #     keys=image_keys,
    #     lower=0.5,
    #     upper=99.5,
    #     b_min=0,
    #     b_max=1,
    #     clip=True,
    #     relative=False
    # ),
    rescaler=ScaleIntensityRangeD(
        keys=image_keys,
        a_min=11,
        a_max=1387,
        b_min=0,
        b_max=1,
        clip=True,
    ),
    resizer=CropForegroundD(
        keys=image_keys+mask_keys,
        source_key='mask',
        margin=50,
    ),
    cropper=None,
    caster=CastToTyped(keys=image_keys, dtype=np.float32),
    to_tensor=None,
    is_supervised=True,
    dataset_type=Dataset,
    additional_transforms=None,
)

for data in tqdm(dataset):
    affine = data['image_meta_dict']['affine']
    output_path = check_dir(out_dir/Path(data['image_meta_dict']['filename_or_obj']).parent.name)

    nib.save( nib.Nifti1Image(data['image'].squeeze(), affine), output_path/str(Path(data['image_meta_dict']['filename_or_obj']).name).replace('.nii', '.nii.gz') )
    nib.save( nib.Nifti1Image(data['mask'].squeeze(), affine), output_path/str(Path(data['mask_meta_dict']['filename_or_obj']).name).replace('.nii', '.nii.gz') )

#%% prepare data json
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from ..utils import get_items_from_file

malignance_path = Path(r'/homes/clwang/Data/jsph_lung/SHIXINGJIEJIE/malignant-adenocarcinoma/Solid')
benign_path = Path(r'/homes/clwang/Data/jsph_lung/SHIXINGJIEJIE/benign')
root_path = Path(r'/homes/yliu/Data/pn_cls_data/MVI/MVI-Benign-Crops-Norm')
B_subdirs = list(filter(lambda x: not str(x.name).endswith('repeat') and not str(x.name).endswith('noboceng'), benign_path.iterdir()))
M_casedirs = list((malignance_path/'MVI-NEGATIVE').iterdir()) + list((malignance_path/'MVI-POSITIVE').iterdir())
datalist = []

per5_list, per995_list = [], []
for subdir in B_subdirs:
    subsubdirs = list(filter(lambda x: x.is_dir(), subdir.iterdir()))
    for case_dir in subsubdirs:
        img_path = str(list(case_dir.glob('*_src.nii'))[0])
        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()
        minv, maxv = np.min(img_data), np.max(img_data)
        if minv < -2000:
            per5, per995 = np.percentile(img_data[img_data>-2000], 0.5), np.percentile(img_data[img_data>-2000], 99.5)
        else: 
            per5, per995 = np.percentile(img_data, 0.5), np.percentile(img_data, 99.5)

        per5_list.append(per5)
        per995_list.append(per995)
        print(minv, maxv, per5, per995)
        
for case in M_casedirs:
    img_path = str(list(case.glob('*lung.nii'))[0])
    img_nii = nib.load(img_path)
    img_data = img_nii.get_fdata()
    img_data = img_data-1024
    minv, maxv = np.min(img_data), np.max(img_data)
    if minv < -2000:
        per5, per995 = np.percentile(img_data[img_data>-2000], 0.5), np.percentile(img_data[img_data>-2000], 99.5)
    else: 
        per5, per995 = np.percentile(img_data, 0.5), np.percentile(img_data, 99.5)
    
    per5_list.append(per5)
    per995_list.append(per995)


#%%

M_datalist = []
for case in M_casedirs:
    case_dict = {}
    case_dict['image'] = str(list(case.glob('*lung.nii'))[0])
    case_dict['mask'] = str(list(case.glob('*roi.nii'))[0])
    case_dict['label'] = 1
    M_datalist.append(case_dict)

B_datalist = []
for subdir in B_subdirs:
    subsubdirs = list(filter(lambda x: x.is_dir(), subdir.iterdir()))
    for case_dir in subsubdirs:
        try:
            case_dict = {}
            case_dict['image'] = str(list(case_dir.glob('*_src.nii'))[0])
            case_dict['mask'] = str(list(case_dir.glob('*_Lesion.nii'))[0])
            case_dict['label'] = 0
            B_datalist.append(case_dict)
        except:
            print('Missing Benign', case_dir.name)

datalist = M_datalist + B_datalist

with open('/homes/yliu/Data/pn_cls_data/SHIXINGJIEJIE/MVI_Benign.json', 'w') as f:
    json.dump(B_datalist, f, indent=2)

with open('/homes/yliu/Data/pn_cls_data/SHIXINGJIEJIE/MVI_Malignance.json', 'w') as f:
    json.dump(M_datalist, f, indent=2)

with open('/homes/yliu/Data/pn_cls_data/SHIXINGJIEJIE/MVI_MB.json', 'w') as f:
    json.dump(datalist, f, indent=2)


#%%
#! split train and test
import json
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from scipy import stats
from sklearn.model_selection import train_test_split

datalist_json = r"\\mega\homesall\clwang\Data\jsph_lung\SHIXINGJIEJIE\crop_data_list_minmax_70.json"
with open(datalist_json) as f:
    datalist = json.load(f)

def count_list(input):
    if not isinstance(input, list):
        input = list(input)
    dict = {}
    for i in set(input):
        dict[i] = input.count(i)
    return dict

def chi_square_unpaired(data1, data2):
    count1 = count_list(data1)
    count2 = count_list(data2)
    print(count1)
    print(count2)

    categories = set(list(count1.keys()) + list(count2.keys()))
    contingency_dict = {}
    for category in categories:
        contingency_dict[category] = [count1[category] if count1[category] else 0, count2[category] if count2[category] else 0]
    contingency_pd = pd.DataFrame(contingency_dict)
    contingency_array = np.array(contingency_pd)
    return stats.chi2_contingency(contingency_array)

def ptest_cat(test_arr1, test_arr2):
    arr1 = np.array(test_arr1)
    arr2 = np.array(test_arr2)
    _, pvalue, _, _ =  chi_square_unpaired(arr1, arr2)
    return pvalue

def split_image_label(data_list):
    X = []
    y = []
    for data in datalist:
        label_ = int(data['label'])
        X.append(data['image'])
        y.append(label_)
    X = np.array(X)
    y = np.array(y)
    return X, y    

def node_wise_datasplit(datalist, ratios):
    x, y = split_image_label(datalist)

    pvalue_th = 0.05
    for ratio in ratios:
        split_raito = ('-'.join(map(lambda x: str(int(x*10)), ratio)))
        for randomseed in range(10000):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio[-1], random_state=randomseed, shuffle=True)
            pvalue = ptest_cat(y_train, y_test)
            if pvalue > pvalue_th:
                train_data = []
                test_data = []
                for img, l in zip(X_train, y_train):
                    if img.endswith('lung.nii.gz'):
                        mask = img.replace('lung.nii.gz', 'roi.nii.gz')
                    else:
                        mask = img.replace('src.nii.gz', 'Lesion.nii.gz')
                    train_data.append({'image': img, 'mask': mask, 'label': int(l)})
                for img, l in zip(X_test, y_test):
                    if img.endswith('lung.nii.gz'):
                        mask = img.replace('lung.nii.gz', 'roi.nii.gz')
                    else:
                        mask = img.replace('src.nii.gz', 'Lesion.nii.gz')
                    test_data.append({'image': img, 'mask': mask, 'label': int(l)})
                
                with open(r"\\mega\yliu\Data\pn_cls_data\MVI\new_for_clwang\train_datalist_{}_MVI_MB.json".format(split_raito), 'w') as f:
                    json.dump(train_data, f, indent=2)
                with open(r"\\mega\yliu\Data\pn_cls_data\MVI\new_for_clwang\test_datalist_{}_MVI_MB.json".format(split_raito), 'w') as f:
                    json.dump(test_data, f, indent=2)
                break

node_wise_datasplit(datalist, ratios=[(0.8, 0.2)])


#%%
import json
from pathlib import Path

# create json
root_path = Path('/homes/yliu/Data/pn_cls_data/SHIXINGJIEJIE/Crop-Norm-minmax')
case_list = list(filter(lambda x: x.is_dir(), root_path.iterdir()))

caselist = []
for case in case_list:
    case_dict = {}
    case_dict['image'] = str((list(case.glob('*lung.nii.gz')) + list(case.glob('*src.nii.gz')))[0])
    case_dict['mask'] = str((list(case.glob('*roi.nii.gz')) + list(case.glob('*Lesion.nii.gz')))[0])
    case_dict['label'] = 1 if case_dict['mask'].endswith('roi.nii.gz') else 0
    caselist.append(case_dict)

with open(root_path/'caselist_crop_norm.json', 'w') as f:
    json.dump(caselist, f, indent=2)
# %%  Plot lidc histogram
from pathlib import Path 
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt

lidc_crop_dir = Path(r"\\mega\homesall\clwang\Data\LIDC-IDRI-Crops-Norm\data")
for dirs in tqdm(lidc_crop_dir.iterdir()):
    for subdir in dirs.iterdir():
        img_nii = nib.load(subdir/'ct_crops.nii.gz')
        img_data = img_nii.get_fdata()
        plt.hist(img_data.ravel(), alpha=0.1)
plt.show()

#%%
import tqdm
from pathlib import Path
from scipy.ndimage import label, generate_binary_structure
from ..utils import get_items_from_file
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

json_path = Path("/homes/clwang/Data/jsph_lung/SHIXINGJIEJIE/data_list.json")
datalist = get_items_from_file(json_path, format='json')

r_list, c_list, z_list = [], [], []
multinode_case = []
for case in tqdm.tqdm(datalist):
    label_path = case['mask'] 
    label_arr = nib.load(label_path).get_fdata()
    rmin, rmax, cmin, cmax, zmin, zmax = bbox_3D(label_arr)
    ccs, num_features = label(label_arr, structure=generate_binary_structure(3,1))
    component_sizes = np.bincount(ccs.ravel())

    print('{} :{} components found!'.format(Path(label_path).parent.name, len(component_sizes)-1))
    if len(component_sizes)-1:
        multinode_case.append(Path(label_path).parent.name)
    if rmax-rmin > 150:
        print('r too large:', Path(label_path).parent.name)
    if cmax-cmin > 150:
        print('c too large:', Path(label_path).parent.name)
    if zmax-zmin > 50:
        print('z too large:', Path(label_path).parent.name)
    # r_list.append(rmax-rmin)
    # c_list.append(cmax-cmin)
    # z_list.append(zmax-zmin)
    
print(np.min(r_list), np.max(r_list), np.min(c_list), np.max(c_list), np.min(z_list), np.max(z_list))

# %%
