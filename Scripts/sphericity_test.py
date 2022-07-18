#%%
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from radiomics.shape import RadiomicsShape
from utils_cw import get_items_from_file

train_json_path = Path("/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/test_datalist_8-2_cls-new.json")
train_list = get_items_from_file(train_json_path, format='json')

all_sphericity = []
result = {'patientID':[], 'sphericity':[], 'label':[]}
case_list = []
for train_data in train_list:
    case_dict = {}
    inputs_path = train_data['image']
    masks_path = train_data['mask']
    features = train_data['features']
    inputs = sitk.ReadImage(inputs_path)
    masks = sitk.ReadImage(masks_path)
    masks.CopyInformation(inputs)
    case_dict['image'] = train_data['image']
    case_dict['mask'] = train_data['mask']
    case_dict['label'] = train_data['label']

    RadiomicsShape_ = RadiomicsShape(inputImage=inputs, inputMask=masks)
    get_sphericity = RadiomicsShape_.getSphericityFeatureValue()
    result['patientID'].append(str(Path(inputs_path).parent.parent.name)+'_'+str(Path(inputs_path).parent.name))
    result['sphericity'].append(f'{get_sphericity:.3f}')
    result['label'].append(features[3])
    sphericity = 1 if get_sphericity >= 0.8 else 0
    all_sphericity.append(sphericity)

    case_dict['features'] = train_data['features'][:3] + [sphericity] + train_data['features'][4:]
    case_list.append(case_dict)

print('label 1:', np.sum(all_sphericity))
pd.DataFrame(result).to_csv('/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/sphericity_radiomics-test-new.csv')

with open(train_json_path.parent/'test_datalist_8-2_cls-new-radiomics.json', 'w') as f:
    json.dump(case_list, f, indent=2)

#%%
import json
from pathlib import Path
from utils_cw import get_items_from_file

root_path = Path(r"/homes/yliu/Data/pn_cls_data/LIDC-IDRI/normed/train_datalist_8-2_cls-new-radiomics.json")
datalist = get_items_from_file(root_path, format='json')

