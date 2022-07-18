## Official repo of the paper "Towards Reliable and Explainable AI Model for Pulmonary Nodule Diagnosis"
---

### For the model training:
`python main.py --param-path /your/parameters_file.json` \
An simple example of parameter file is shown in [param.json](param.json).

### For the model testing:
`python test.py` \
Notice: change the hard-coded variable `model_path` and `test_json` for your project. 

### Requirements
You need to pre-install the following packages for this program:
- pytorch
- tb-nightly
- click
- tqdm
- numpy
- scipy
- scikit-learn
- nibabel
- pytorch-ignite
- [strix](https://github.com/Project-Strix/Strix)
- [monai_ex](https://github.com/Project-Strix/MONAI_EX)
- [utils_cw](https://gitlab.com/ChingRyu/py_utils_cw)