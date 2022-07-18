#%%
from enum import EnumMeta
import json
from pathlib import Path
from matplotlib.pyplot import get
import pandas as pd
import numpy as np
from utils_cw import get_items_from_file

label = '侵犯'  #! 232/78
features = ['毛刺征', '分叶征', '空泡', '周围模糊影']
##! 250/144  45/348 273/121  274/120
final_version_path = Path("/homes/clwang/Data/jsph_lung/MVI/data/脉管侵犯MVI_Final version.xlsx")
first_version_path = Path("/homes/yliu/Code/jsph_mvi_cls/MVI_Datafile_for_deeplearning.xlsx")
json_path = Path("/homes/clwang/Data/jsph_lung/MVI/data_crops/datalist-test.json")

df = pd.read_excel(final_version_path)
first_df = pd.read_excel(first_version_path, header=2)

json_file = get_items_from_file(json_path)
case_list = []
for case_dict in json_file:
    case_id = str(Path(case_dict['image']).parent.name)
    _label = 1 if 'POSITIVE' in str(Path(case_dict['image']).parent.parent.name) else 0
    if case_id in list(map(lambda x: str(x), df['CaseName'].to_list())):
        if case_id.startswith('0') and case_id.isdigit():
            case_id = int(case_id[3:])
        elif case_id.isdigit():
            case_id = int(case_id)
        _features = list(map(lambda x: int(x), list(df.loc[df['CaseName']==case_id, features].values.squeeze())))
        if len(_features) == 4:
            case_dict['label'] = int(_label)
            case_dict['features'] = list(_features)
            case_list.append(case_dict)
        else:
            print('not in final', case_id, len(_features), len(_label))
    elif case_id in list(map(lambda x: str(x), first_df['Filename'].to_list())):
        try:
            name = first_df.loc[first_df['Filename']==case_id, 'Name'].item()
        except:
            name = first_df.loc[first_df['Filename']==int(case_id), 'Name'].item()
        _features = list(map(lambda x: int(x), list(df.loc[df['姓名']==name, features].values.squeeze())))
        if len(_features) == 4:
            case_dict['label'] = int(_label)
            case_dict['features'] = list(_features)
            case_list.append(case_dict)
        else:
            print('not in first', case_id, len(_features), len(_label))
    else:
        print('not find', case_id)


with open('/homes/yliu/Code/jsph_mvi_cls/datalist-withfeat-test.json', 'w') as f:
    json.dump(case_list, f, indent=2)

#%%
#! add features 
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from utils_cw import get_items_from_file
from utils import binarization


ROOT_PATH = Path('/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data')
SAVE_PATH = Path("/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed")
train_json = Path("/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/train_datalist_8-2_cls.json")
test_json = Path("/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/test_datalist_8-2_cls.json")

# FEATURE_KEY = ['Subtlety','Calcification','Sphericity',
#                 'Margin', 'Texture']
## 文献精度(AUC/accuracy) 0.803/0.719 0.930/0.908 0.568/0.552 0.776/0.725 0.850/0.834 

FEATURE_KEY = ['Subtlety','Calcification', 'Texture','Sphericity',
                'Margin']
                # , 'Lobulation', 'Spiculation']
## 0.803/0.719 0.930/0.908 0.850/0.834  0.568/0.552 0.776/0.725
## 0.79        0.93        0.94         0.69        0.82

##! normed 良恶性(0:1/895:400)
## 去除只有1个医生标注的数据
#! 1-3(245/700) / 1-5(211/734)/ 1-3(279/666)/ 1-3(215/730)/ 1-3(101/844)
#! [0.35, 0.28, 0.42, 0.30, 0.12]
#! (54/191) (67/178) (62/183) (53/192) (15/230)

#* 去除只有1个医生标注的数据 新
#* train 1-3(299/681) 1-5(213/767) 1-3(82/898) 1-3(277/703) 1-3(194/786)
#* [0.43, 0.27, 0.1, 0.4, 0.25]
#* test (71/175) (72/174) (19/227) (65/181) (45/201)
#* 球状度 (546/434) 1.2 (134/112)


def round_off(number):
    number = int(number+0.5)
    return number

train_list = get_items_from_file(train_json)
test_list = get_items_from_file(test_json)
case_list =[]
feature_List = []
for train_data in test_list:
    image_path = train_data['image']
    attribute_path = Path(image_path).parent/'attributes.csv'
    attribute_df = pd.read_csv(attribute_path)
    if attribute_df.shape[0] < 2:
        continue
    else:
        feature_list = []
        for _feature in FEATURE_KEY:
            feat_list = attribute_df[_feature].to_list()
            # collection_words = Counter(feat_list).most_common(1)[0]
            # maxlabel = collection_words[0] if collection_words[1]>1 else int(np.median(feat_list))
            maxlabel = np.mean(feat_list)
            if _feature == 'Calcification':
                maxlabel = 1 if maxlabel > 5 else 0
            elif _feature in ['Lobulation', 'Spiculation']:
                maxlabel = 1 if maxlabel > 1 else 0
            else :
                maxlabel = 1 if maxlabel > 3 else 0
            feature_list.append(maxlabel)
        feature_List.append(feature_list)
        train_data['features'] = feature_list
        case_list.append(train_data)

with open(SAVE_PATH/'test_datalist_8-2_cls-new.json', 'w') as f:
    json.dump(case_list, f, indent=2)
feat_arr = np.array(feature_List)
feat_num = feat_arr.shape[1]
for num in range(feat_num):
    _feat_arr = list(feat_arr[:,num])
    collection_words = Counter(_feat_arr).most_common(5)
    print(collection_words)

# %% Generate Datalist
import csv, json, math
import numpy as np
import pandas as pd
from pathlib import Path

def round_off(number):
    number = int(number+0.5)
    return number

root_dir = Path(r'/homes/clwang/Data/LIDC-IDRI-Crops-Norm/data')
save_dir = Path(r'/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed')
datalist = []
for crop in root_dir.rglob('ct_crops.nii.gz'):
    csvfile = crop.with_name('attributes.csv')
    with csvfile.open() as f:
        reader = csv.DictReader(f, delimiter=',')
        scores = [int(row['Malignancy']) for row in reader]
        malignancy = np.mean(scores)
        #! remove only one attributes
        if len(scores) < 2:
            continue
        else:
            # malignancy = Counter(scores).most_common(1)[0][0]
            datalist.append(
                {'image':str(crop), 'mask':str(crop.with_name('consensus_msk_crops.nii.gz')), 'label':malignancy}
            )

with (save_dir/'all_datalist.json').open('w') as f:
    json.dump(datalist, f, indent=2)

#%%
#! split train and test
import json
from pathlib import Path
import pandas as pd
import nibabel as nib
from scipy import stats
from sklearn.model_selection import train_test_split

datalist_json = "/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/all_datalist.json"
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
        if label_ != 3: #! remove label 3.
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

                    train_data.append({'image': img, 'mask': img.replace('ct_crops', 'consensus_msk_crops'), 'label': int(l)})
                for img, l in zip(X_test, y_test):
                    test_data.append({'image': img, 'mask': img.replace('ct_crops', 'consensus_msk_crops'), 'label': int(l)})
                with open(datalist_json.replace('all_datalist', f'train_datalist_{split_raito}_cls'), 'w') as f:
                    json.dump(train_data, f, indent=2)
                with open(datalist_json.replace('all_datalist', f'test_datalist_{split_raito}_cls'), 'w') as f:
                    json.dump(test_data, f, indent=2)
                break

node_wise_datasplit(datalist, ratios=[(0.8, 0.2)])

# %%
from pathlib import Path
from utils_cw import get_items_from_file

train_path = Path("/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/RM1-featworound/train_datalist_8-2_cls-new-radiomics.json")
test_path = Path("/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/RM1-featworound/test_datalist_8-2_cls-new-radiomics.json")
case_list = []

node_path_list1 = get_items_from_file(train_path, format='json')
node_path_list2 = get_items_from_file(test_path, format='json')
node_path_list = node_path_list1 + node_path_list2
for node_path in node_path_list:
    case_name = Path(node_path['image']).parent.parent.name
    if case_name not in case_list:
        case_list.append(case_name)
# %%
import sys
sys.path.append(r'\\mega\yliu\Code\utils_yliu')
import json
from pathlib import Path
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics.shape import RadiomicsShape
from scipy import stats
from monai.transforms import LoadImageD, AddChannelD, SpacingD, LambdaD, Compose
from monai.utils.enums import GridSampleMode

FEATURE_KEY = ['Subtlety', 'Calcification', 'Texture', 'Margin', 'Sphericity']
train_json = Path(r"\\mega\homesall\clwang\Data\LIDC-IDRI-Crops-Norm\data-minmax\train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")
test_json = Path(r"\\mega\homesall\clwang\Data\LIDC-IDRI-Crops-Norm\data-minmax\test_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")
save_path = Path(r'\\mega\yliu\Data\pn_cls_data\LIDC-IDRI')

train_list = get_items_from_file(train_json, format='json')
test_list = get_items_from_file(test_json, format='json')

trans = Compose([
        LoadImageD(keys=['image', 'mask']),
        AddChannelD(keys=['image', 'mask']),
        SpacingD(keys=['image', 'mask'], pixdim=(0.7, 0.7, 1.25), mode=[GridSampleMode.BILINEAR, GridSampleMode.BILINEAR]),
        LambdaD(keys='mask', func=lambda x: x>0),
])

all_case = []
feature_List = []
raw_sphericity = []
for _case in test_list:
    img_path = _case['image'].replace('/homes','\\\\mega\\homesall').replace('/','\\')
    msk_path = _case['mask'].replace('/homes','\\\\mega\\homesall').replace('/','\\')
    attribute_csv = Path(img_path).parent/'attributes.csv'
    features_df = pd.read_csv(attribute_csv, header=0)

    _raw_sphericity = 1 if np.mean(features_df['Sphericity'].to_list()) > 3 else 0
    raw_sphericity.append(_raw_sphericity)
    
    ret = trans({'image': img_path, 'mask': msk_path})
    img_arr = ret['image'].squeeze()
    msk_arr = ret['mask'].astype(np.int).squeeze()

    inputs = sitk.GetImageFromArray(img_arr)
    masks = sitk.GetImageFromArray(msk_arr)
    masks.CopyInformation(inputs)

    RadiomicsShape_ = RadiomicsShape(inputImage=inputs, inputMask=masks)
    get_sphericity = RadiomicsShape_.getSphericityFeatureValue()

    feature_list = []
    for _feature in FEATURE_KEY[:-1]:
        feat_list = features_df[_feature].to_list()
        maxlabel = np.mean(feat_list)
        if _feature == 'Calcification':
            maxlabel = 1 if maxlabel > 5 else 0
        else :
            maxlabel = 1 if maxlabel > 3 else 0
        feature_list.append(maxlabel)
    sphericity = 1 if get_sphericity >= 0.8 else 0
    feature_list.append(sphericity)
    _case['features'] = feature_list
    feature_List.append(feature_list)
    all_case.append(_case)

with open(save_path/'test_raw_Sphericity.json', 'w') as f:
    json.dump(raw_sphericity, f, indent=2)
with open(save_path/test_json.name, 'w') as f:
    json.dump(all_case, f, indent=2)
# %%
#!  检验train和test间特征的显著性差异
#!  检验radiomics计算的球状度和原始的球状度之间的一致性
import sys
sys.path.append(r'\\mega\yliu\Code\utils_yliu')
from pathlib import Path
import pandas as pd
from utils_cw import get_items_from_file
from StatisticalAna import P_test, chi_square_unpaired
from sklearn.metrics import cohen_kappa_score

FEATURE_KEY = ['Subtlety', 'Calcification', 'Texture', 'Margin', 'Sphericity', 'label']
train_sphericity = Path(r"\\mega\yliu\Data\pn_cls_data\LIDC-IDRI\train_raw_Sphericity.json")
test_sphericity = Path(r"\\mega\yliu\Data\pn_cls_data\LIDC-IDRI\test_raw_Sphericity.json")

train_feat_json = Path(r"\\mega\yliu\Data\pn_cls_data\LIDC-IDRI\train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")
test_feat_json = Path(r"\\mega\yliu\Data\pn_cls_data\LIDC-IDRI\test_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")

raw_sphericity = get_items_from_file(train_sphericity, format='json') +  get_items_from_file(test_sphericity, format='json')
train_json = get_items_from_file(train_feat_json, format='json')
test_json = get_items_from_file(test_feat_json, format='json')
train_feat = {key:[] for key in FEATURE_KEY}
test_feat = {key:[] for key in FEATURE_KEY}

for train_case in train_json:
    features = train_case['features']
    for i, _feat in enumerate(features):
        train_feat[FEATURE_KEY[i]].append(_feat)
    train_feat[FEATURE_KEY[-1]].append(train_case['label'])
for test_case in test_json:
    features = test_case['features']
    for i, _feat in enumerate(features):
        test_feat[FEATURE_KEY[i]].append(_feat)
    test_feat[FEATURE_KEY[-1]].append(test_case['label'])

train_feat_arr = pd.DataFrame(train_feat)
test_feat_arr = pd.DataFrame(test_feat)
radiomics_sphericity = list(train_feat_arr['Sphericity']) + list(test_feat_arr['Sphericity'])
kappa =  cohen_kappa_score(raw_sphericity, radiomics_sphericity)
p_test_result = P_test(train_feat_arr, test_feat_arr, [], FEATURE_KEY)
pd.DataFrame(p_test_result).to_csv(train_feat_json.parent/'ptest1.csv')
# %%

last_radiomics_train = Path(r"\\mega\homesall\clwang\Data\LIDC-IDRI-Crops-Norm\data-minmax\train_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")
last_radiomics_test = Path(r"\\mega\homesall\clwang\Data\LIDC-IDRI-Crops-Norm\data-minmax\test_datalist_8-2_minmax_remove3(ver3)_feature_sphericity.json")

last_train = get_items_from_file(last_radiomics_train, format='json')
last_test = get_items_from_file(last_radiomics_test, format='json')
last_train_sph = []
last_test_sph = []
for _traincase in last_train:
    last_train_sph.append(_traincase['features'][-2])
for _testcase in last_test:
    last_test_sph.append(_testcase['features'][-2])

last_radiomics = last_train_sph + last_test_sph
kappa_1 =  cohen_kappa_score(last_radiomics, radiomics_sphericity)
kappa_2 =  cohen_kappa_score(last_radiomics, raw_sphericity)
# %%
