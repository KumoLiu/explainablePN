#%%
import json, tqdm
import pandas as pd
from pathlib import Path
from scipy.ndimage.measurements import label
from utils_cw import get_items_from_file

root_label_path = Path(r"\\mega\yliu\Data\pn_cls_data\MPS\MPL及SOL excel 表格修改0208.xlsx")
train_json = Path(r"\\mega\homesall\clwang\Data\jsph_lung\YHBLXA_YXJB\data_crops\datalist-train-yyq.json")
test_json = Path(r"\\mega\homesall\clwang\Data\jsph_lung\YHBLXA_YXJB\data_crops\datalist-test-yyq.json")
error_json = Path(r"\\mega\homesall\clwang\Data\jsph_lung\YHBLXA_YXJB\error_dict.json")
SAVE_PATH = Path(r'\\mega\yliu\Data\pn_cls_data\MPS')

refer_train_path = Path(r"\\mega\yliu\Data\medlp_exp\classification\jsph_mps\resnet50-BCE-BN-sgd-plateau-0622_1914-lr0.1-CTRList\train_files.yml")
refer_valid_path = Path(r"\\mega\yliu\Data\medlp_exp\classification\jsph_mps\resnet50-BCE-BN-sgd-plateau-0622_1914-lr0.1-CTRList\valid_files.yml")

_FEATURE_KEY = ['lobulation', 'spiculation', 'Relation to bronchus', 'Relation to Vessel']
FEATURE_KEY = [
        'lobulation（0-no；1-shallow lobulation；2-deep lobulation）', 
        'spiculation', 'Relation to bronchus（Negative（0）：0+3+4+5 ；Postive（1）：1+2）', 
        'Relation to Vessel（0；type Ⅰ ；type Ⅱ；type Ⅲ；type Ⅳ）'
        # 'Relation to Vessel（Negative（0）：0+1+2 ；Postive（1）：3+4）'
        ]
#! train: 296/364 484/177 429/232 464/197  label 421/240
#! test:  65/86 108/43 97/54 109/42

#!new:
#! train: 246/282 387/141 343/185 373/155   label 336/192
#! valid: 55/78 97/36 85/48 91/42

refer_train_list = get_items_from_file(refer_train_path, format='yaml')
refer_valid_list = get_items_from_file(refer_valid_path, format='yaml')
train_list = get_items_from_file(train_json, format='json')
test_list = get_items_from_file(test_json, format='json')
error_dict = get_items_from_file(error_json, format='json')
label_df = pd.read_excel(root_label_path, header=2, index_col=0)
label_df['Clinc ID'] = list(map(lambda x: str(x).upper(), label_df['Clinc ID'].tolist()))
error_id_list = list(error_dict.keys())
case_list = []
check_id = []
feature_List = []
for case in tqdm.tqdm(refer_train_list):
    case_id = Path(case['image']).parent.name
    if case_id == '0007675936_little':
        case_id = case_id.upper()
    
    if case_id in error_id_list:
        indices = [case_id in item for item in label_df['Clinc ID']]
        indices = indices if sum(indices) == 1 else [case_id == item for item in label_df['Clinc ID']]
        case_id = case_id if sum(indices) == 1 else error_dict[case_id]
    indices = [case_id in item or case_id.strip('0') in item for item in label_df['Clinc ID']]
    indices = indices if sum(indices) == 1 else [case_id == item for item in label_df['Clinc ID']]
    if sum(indices) > 1:
        print('****Multiple cases:', case_id)

    try:
        feat_list = [label_df.loc[indices, feat].item() for feat in FEATURE_KEY]
        case['features'] = feat_list
        case_list.append(case)
        feature_List.append(feat_list)
    except:
        check_id.append(case_id)
        # print(case_id)

# with open(SAVE_PATH/'crop'/'datalist-train-mps_newf4.json', 'w') as f:
#     json.dump(case_list, f, indent=2)

print(check_id)

#%%
import numpy as np
from collections import Counter

feat_arr = np.array(feature_List)
feat_num = len(FEATURE_KEY)
for num in range(feat_num):
    _feat_arr = list(feat_arr[:,num])
    collection_words = Counter(_feat_arr).most_common(5)
    print(collection_words)

#%%
from pathlib import Path
from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np

root_path = Path('/homes/clwang/Data/jsph_lung/YHBLXA_YXJB/data_crops')
seg_path_list = list(root_path.glob('*/*/*/*Lesion.nii.gz'))

def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax

z_list = []
for seg_path in seg_path_list:
    seg_arr = nib.load(seg_path).get_fdata()
    edge = bbox_3D(seg_arr)
    z_list.append(edge[-1]-edge[-2])
    if edge[-1]-edge[-2] > 50:
        print(seg_path.parent.name)

#%%
#!  检验train和test间特征的显著性差异
import sys
sys.path.append(r'\\mega\yliu\Code\utils_yliu')
from pathlib import Path
import pandas as pd
from utils_cw import get_items_from_file
from StatisticalAna import P_test, chi_square_unpaired

FEATURE_KEY = ['lobulation', 'spiculation', 'Relation to bronchus', 'Relation to Vessel', 'label']

train_feat_json = Path(r"\\mega\yliu\Data\pn_cls_data\MPS\crop\datalist-train-mps_new_addprob.json")
valid_feat_json = Path(r"\\mega\yliu\Data\pn_cls_data\MPS\crop\datalist-valid-mps_new_addprob.json")
test_feat_json = Path(r"\\mega\yliu\Data\pn_cls_data\MPS\crop\datalist-test-mps_new_addprob.json")

train_json = get_items_from_file(train_feat_json, format='json')
valid_json = get_items_from_file(valid_feat_json, format='json')
test_json = get_items_from_file(test_feat_json, format='json')
train_feat = {key:[] for key in FEATURE_KEY}
test_feat = {key:[] for key in FEATURE_KEY}

for train_case in train_json:
    features = train_case['features']
    for i, _feat in enumerate(features):
        train_feat[FEATURE_KEY[i]].append(_feat)
    train_feat[FEATURE_KEY[-1]].append(train_case['label'])
for valid_case in valid_json:
    features = valid_case['features']
    for i, _feat in enumerate(features):
        train_feat[FEATURE_KEY[i]].append(_feat)
    train_feat[FEATURE_KEY[-1]].append(valid_case['label'])
for test_case in test_json:
    features = test_case['features']
    for i, _feat in enumerate(features):
        test_feat[FEATURE_KEY[i]].append(_feat)
    test_feat[FEATURE_KEY[-1]].append(test_case['label'])

train_feat_arr = pd.DataFrame(train_feat)
test_feat_arr = pd.DataFrame(test_feat)
p_test_result = P_test(train_feat_arr, test_feat_arr, [], FEATURE_KEY)
pd.DataFrame(p_test_result).to_csv(train_feat_json.parent/'ptest.csv')
