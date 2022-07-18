import time, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import numpy as np
from pathlib import Path
from monai.data import DataLoader
from monai.metrics import compute_roc_auc
from monai.transforms import AsDiscrete, Activations
from datasets import CLASSIFICATION_DATASETS
from sklearn.metrics import classification_report
from utils import DrawROCList, get_network, check_dir, get_items_from_file
from metrics import AUC_Confidence_Interval, save_roc_curve_fn

model_path = Path("/homes/yliu/Data/pn_cls_exp/lidc-paper_new-sphericity_prob/1215_2358-raw_hesam_agg-slice_1-lr_0.01-plateau-multi-BCE-sgd-sum-parallel-wooffset-2nd/Models/SBAA/BestModel@42with12.817.pt")
config_path = model_path.parent.parent.parent/'param.list'
configures = get_items_from_file(config_path, format='json')

IMAGE_KEY = 'image'
LABEL_KEY = 'label'
if 'prob' in configures['dataset_name']:
    MASK_KEY = 'probmap'
elif 'seg' in configures['dataset_name']:
    MASK_KEY = 'seg'
else: 
    MASK_KEY = 'mask'
FEATURE_KEY = None #'features' # 
preload = 0
device = torch.device("cuda")


test_json = Path("/homes/yliu/Data/pn_cls_data/LIDC-IDRI/all_datalist_int_equals3.json")
dataset_name = configures['dataset_name']
dimensions = configures['dimensions']
dataset_type = CLASSIFICATION_DATASETS[f'{dimensions}D'][dataset_name]['FN']
files_test = get_items_from_file(test_json)
mode = configures['mode']
net = configures['net']
features = configures['features']
in_channels = configures['in_channels']
out_channels = configures['out_channels']
out_channels_f = configures['out_channels_f']
test_ds = dataset_type(files_test, 'test', {'input_nc': in_channels,'output_nc': out_channels, 'preload': preload})
test_loader = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=10)
test_save_path = check_dir(model_path.parent.parent/f'{time.strftime("%m%d_%H%M")}-{model_path.name}')
save_latent = False
save_to_np = False

model = get_network(
    name=net,
    dimensions=dimensions,
    in_channels=in_channels,
    out_channels=out_channels,
    out_channels_f=out_channels_f,
    features=features,
    device=device,
    mode=mode,
    # save_attentionmap_fpath=str(test_save_path),
    save_attentionmap_fpath=None,
    use_attention=False,
    use_cbam=False,
    use_mask=False,
    use_aspp=False,
    save_latent=save_latent,
)
model.load_state_dict(torch.load(model_path))
model.eval()

y_pred = []
patientID = []
result = {'label_auc': [], 'label_auc_CI':[], 'feat_auc': [], 'feat_auc_CI': []}
result_wofeat = {'label_auc': [], 'label_auc_CI':[]}
latent_codes_MB, latent_codes_feat = torch.tensor([], device=device), torch.tensor([], device=device)
with torch.no_grad():
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    y_feat_pred = torch.tensor([], dtype=torch.float32, device=device)
    y_feat = torch.tensor([], dtype=torch.long, device=device)
    for test_data in test_loader:
        test_images, test_labels, test_masks, _patientID = (
            test_data[IMAGE_KEY].to(device),
            test_data[LABEL_KEY].to(device),
            test_data[MASK_KEY].to(device).float(),
            [Path(a).parent.name for a in test_data['image_meta_dict']['filename_or_obj']]
        )
        patientID += _patientID

        if FEATURE_KEY is not None:
            test_features = test_data[FEATURE_KEY].to(device)
        else:
            test_features = None

        test_outputs= model(test_images, test_labels, test_features, test_masks)
        if save_latent:
            if len(test_outputs) == 2:
                if len(test_outputs[1].shape) == 1:
                    test_outputs[1] = test_outputs[1].unsqueeze(0)
                latent_codes_MB = torch.cat([latent_codes_MB, test_outputs[1]], dim=0)
            elif len(test_outputs[2]) > 1:
                if len(test_outputs[2][0].shape) == 1:
                    test_outputs[2][0] = test_outputs[2][0].unsqueeze(0) 
                    test_outputs[2][1] = test_outputs[2][1].unsqueeze(0) 
                latent_codes_MB = torch.cat([latent_codes_MB, test_outputs[2][0]], dim=0)
                latent_codes_feat = torch.cat([latent_codes_feat, test_outputs[2][1]], dim=0)
            else:
                if len(test_outputs[2].shape) == 1:
                    test_outputs[2] = test_outputs[2].unsqueeze(0) 
                latent_codes_MB = torch.cat([latent_codes_MB, test_outputs[2]], dim=0)

        test_output1 = test_outputs[0]
        if FEATURE_KEY is not None:
            test_output1 = test_outputs[0]
            test_output2 = test_outputs[1]
            if len(test_output2.shape) == 1:
                test_output2 = test_output2.unsqueeze(0)
            y_feat_pred = torch.cat([y_feat_pred, test_output2], dim=0)
            y_feat = torch.cat([y_feat, test_features], dim=0)

        if len(test_output1.shape) == 1:
            test_output1 = test_output1.unsqueeze(0)
        y_pred = torch.cat([y_pred, test_output1], dim=0)
        y = torch.cat([y, test_labels], dim=0)

    
    #! 保存全连接前特征latent
    print('latent_MB', latent_codes_MB.shape)
    if save_to_np:
        np.savez(test_save_path / 'fc_latent_MB.npz', latent_codes_MB.cpu().numpy())
    else:
        torch.save(test_save_path / 'fc_latent_MB.pt', latent_codes_MB.cpu())
    if FEATURE_KEY is not None:
        if save_to_np:
            np.savez(test_save_path / 'fc_latent_feat.npz', latent_codes_feat.cpu().numpy())
        else:
            torch.save(test_save_path / 'fc_latent_feat.pt', latent_codes_feat.cpu())


    if FEATURE_KEY is not None:
        y_feat_pred_act = Activations(sigmoid=True)(y_feat_pred.squeeze())
    y_pred_act = Activations(sigmoid=True)(y_pred)
    auc_metric = compute_roc_auc(y_pred_act, y)
    single_auc, mean_auc, CI, sorted_scores, std_auc = AUC_Confidence_Interval(y.cpu().numpy(), y_pred_act.cpu().numpy().squeeze())
    save_roc_curve_fn(y_pred_act.cpu().numpy(), y.cpu().numpy(),test_save_path,MOD='MB')
    
    if 'jsph' in dataset_name:
        name_list = ['MP/S', 'lobulation', 'spiculation', 'Relation to bronchus', 'Relation to Vessel']
    elif 'lidc' in dataset_name:
        name_list = ['Malignance', 'Subtlety','Calcification', 'Texture','Margin', 'Sphericity']
    
    if FEATURE_KEY is not None:
        #! draw roc 
        all_roc_pred = [y_pred_act.cpu().numpy()] + [y_feat_pred_act[:,i].cpu().numpy() for i in range(y_feat_pred_act.shape[1])]
        all_roc_label = [y.cpu().numpy()] + [y_feat[:,i].cpu().numpy() for i in range(y_feat_pred_act.shape[1])]
        DrawROCList(all_roc_pred, all_roc_label, name_list=name_list, store_path=str(test_save_path/'feat_roc.png'))
    else:
        DrawROCList([y_pred_act.cpu().numpy()], [y.cpu().numpy()], name_list=['Malignance/Benign'], store_path=str(test_save_path/'MB_roc.png'))
    
    if FEATURE_KEY is not None:
        feat_auc = []
        feat_auc_CI = []
        for i in range(y_feat_pred_act.shape[1]):
            _feat_auc = compute_roc_auc(y_feat_pred_act[:,i], y_feat[:,i])
            single_auc, mean_auc, _CI, sorted_scores, std_auc = AUC_Confidence_Interval(y_feat[:,i].cpu().numpy(), y_feat_pred_act[:,i].cpu().numpy())
            save_roc_curve_fn(y_feat_pred_act[:,i].cpu().numpy(), y_feat[:,i].cpu().numpy(),test_save_path,MOD=i)
            feat_auc.append(_feat_auc)
            feat_auc_CI.append(_CI)
        result['feat_auc'].append(feat_auc)
        result['label_auc'].append(auc_metric)
        result['label_auc_CI'].append(CI)
        result['feat_auc_CI'].append(feat_auc_CI)
        pd.DataFrame(result).to_csv(test_save_path/'auc_reault.csv')

        all_prob = torch.cat([y_pred_act, y_feat_pred_act], dim=1)
        feat_prob = {x: [] for x in name_list}
        for i, key in enumerate(list(feat_prob.keys())):
            feat_prob[key] = all_prob[:, i].cpu().numpy()
        feat_prob_df = pd.DataFrame(feat_prob)
        feat_prob_df.insert(0,'caseID',patientID)
        feat_prob_df.insert(1,'label',y.cpu().numpy())
        feat_prob_df.to_csv(test_save_path/'prob.csv', index=False)
    else:
        result_wofeat['label_auc'].append(auc_metric)
        result_wofeat['label_auc_CI'].append(CI)
        pd.DataFrame(result_wofeat).to_csv(test_save_path/'auc_reault.csv')
        feat_prob_df = pd.DataFrame()
        feat_prob_df.insert(0,'caseID',patientID)
        feat_prob_df.insert(1,'label',y.cpu().numpy())
        feat_prob_df.insert(2,'pred',y_pred_act.cpu().numpy())
        feat_prob_df.to_csv(test_save_path/'prob.csv', index=False)


    print('test_auc:', auc_metric)
    if FEATURE_KEY is not None:
        print('feat_auc:', feat_auc)
    test_pred = AsDiscrete(threshold_values=True, logit_thresh=0.5)(y_pred_act)
    test_pred_list = list(test_pred.squeeze().detach().cpu().numpy())
    y_list = list(y.squeeze().detach().cpu().numpy())
    report = classification_report(y_list, test_pred_list, target_names=['subetype0','subetype1'], digits=4)
    print(report)