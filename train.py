import json, math
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import dataset
import tqdm, time, os, yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import numpy as np
from pathlib import Path
from monai.data import DataLoader
from monai.metrics import compute_roc_auc
from monai.transforms import AsDiscrete, Activations 
from datasets import CLASSIFICATION_DATASETS
from sklearn.model_selection import train_test_split
from utils import (
    EarlyStopping,
    PolynomialLRDecay,
    plot_fig,
    Save_best_n_models,
    get_network,
    plot_scores,
    cls_bce_loss,
    DWA_bce_loss,
    cls_ce_loss,
    get_items_from_file,
    check_dir
    )
torch.multiprocessing.set_sharing_strategy('file_system')

param_path = Path.home()/"Code"/"ExPN-Net"/"param.list"
output_root_dir = Path.home()/"Data"/"ExPN-Net"
configures = get_items_from_file(param_path, format='json')

###! data parameters 
IMAGE_KEY = 'image'
LABEL_KEY = 'label'
if 'prob' in configures['dataset_name']:
    MASK_KEY = 'probmap'
elif 'seg' in configures['dataset_name']:
    MASK_KEY = 'seg'
else: 
    MASK_KEY = 'mask'
FEATURE_KEY = 'features'
mode = configures['mode']
seed = configures['seed']
dimensions = configures['dimensions']
early_stop = configures['early_stop']
n_epoch = configures['n_epoch']
features = configures['features']
valid_interval = configures['valid_interval']
dataset_name = configures['dataset_name']
reduction = configures['reduction']
optim = configures['optim']
loss_name = configures['loss_name']
lr_policy = configures['lr_policy']
net = configures['net']
in_channels = configures['in_channels']
out_channels = configures['out_channels']
out_channels_f = configures['out_channels_f']
preload = configures['preload']
batch_size = configures['batch_size']
lr = configures['lr']
pos_weight = configures['pos_weight']
weight = configures['weight']
postfix = configures['postfix']
amp = configures['amp']
device = torch.device("cuda")
save_dir_name = f'{time.strftime("%m%d_%H%M")}-{net}-slice_{in_channels}-lr_{lr}-{lr_policy}-{loss_name}-{optim}-{reduction}-{mode}{postfix}'
out_dir = check_dir(output_root_dir/{dataset_name}/{save_dir_name})
with open(out_dir/'param.list', 'w') as f:
    json.dump(configures, f, indent=2)


###! define dataloader
dataset_type = CLASSIFICATION_DATASETS[f'{dimensions}D'][dataset_name]['FN']
if 'lidc' in dataset_name:
    dataset_list = CLASSIFICATION_DATASETS[f'{dimensions}D'][dataset_name]['PATH']
    datalist = get_items_from_file(dataset_list, format='json')
    files_train, files_valid = train_test_split(datalist, test_size=0.2, random_state=seed)
else:
    files_train_p = output_root_dir/"MPS"/"crop"/"datalist-train-mps_new_addprob.json"
    files_valid_p = output_root_dir/"MPS"/"crop"/"datalist-valid-mps_new_addprob.json"
    files_train = get_items_from_file(files_train_p, format='json')
    files_valid = get_items_from_file(files_valid_p, format='json')
train_ds = dataset_type(files_train, 'train', {'input_nc': in_channels,'output_nc': out_channels, 'preload': preload})
valid_ds = dataset_type(files_valid, 'valid', {'input_nc': in_channels,'output_nc': out_channels, 'preload': preload})

with check_dir(out_dir, 'train_list.yml', isFile=True).open('w') as f:
    yaml.dump(files_train, f)
with check_dir(out_dir, 'valid_list.yml', isFile=True).open('w') as f:
    yaml.dump(files_valid, f)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=10)
    
###! define networks
model = get_network(
    name=net,
    dimensions=dimensions,
    features=tuple(features),
    in_channels=in_channels,
    out_channels=out_channels,
    out_channels_f=out_channels_f,
    device=device,
    mode=mode,
    save_attentionmap_fpath=None,
    use_attention=False,
    use_cbam=False,
    use_mask=False,
    use_aspp=False,
)


if os.path.isfile(configures.get('pretrain_model')):
    print('load pretrain model ....')
    checkpoint = torch.load(configures['pretrain_model'])
    model_dict = model.state_dict().copy()
    filtered_dict = {k: v for k, v in checkpoint.items() if v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

if loss_name == 'DWA-BCE':
    loss_function = cls_bce_loss(pos_weight=pos_weight, weight=weight, reduction=reduction)
elif loss_name == 'BCE':
    loss_function = BCEWithLogitsLoss()
elif loss_name == 'CE':
    loss_function = nn.CrossEntropyLoss()
elif loss_name == 'multi-BCE':
    loss_function = cls_bce_loss(pos_weight=pos_weight, weight=weight, reduction=reduction)
elif loss_name == 'multi-CE':
    loss_function = cls_ce_loss(pos_weight=pos_weight, weight=weight, reduction=reduction)


if optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=0.0001)
elif optim =='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr)
if lr_policy == 'const':
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x:1)
elif lr_policy == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='max',
                                                        factor=0.1,
                                                        patience=15,
                                                        cooldown=50,
                                                        min_lr=1e-5)
elif lr_policy == 'poly':
    lr_scheduler = PolynomialLRDecay(optimizer,
                                    n_epoch, 
                                    end_learning_rate=lr*0.1,
                                    power=0.9)

model_saver_dir_SBA = check_dir(Path(out_dir)/'Models'/'SBA')
model_saver_dir_SBL = check_dir(Path(out_dir)/'Models'/'SBL')
model_saver_dir_SBAA = check_dir(Path(out_dir)/'Models'/'SBAA')
model_saver_SBA = Save_best_n_models(model, model_saver_dir_SBA, n_saved=5)
model_saver_SBL = Save_best_n_models(model, model_saver_dir_SBL, n_saved=5)
model_saver_SBAA = Save_best_n_models(model, model_saver_dir_SBAA, n_saved=5)
early_stopping = EarlyStopping(patience=early_stop, verbose=True, path=model_saver_dir_SBA.parent/'checkpoint.pt')

###! model training 
best_metric = -1
best_metric_epoch = -1
train_iters, valid_iters = [], []
epoch_loss_values = []
valid_loss_values = []
learning_rate = []
metric_values= []
epoch_feat_auc = []
for epoch in range(n_epoch):
    model.train()
    epoch_loss = 0
    epoch_loss_MB = 0
    epoch_loss_feature = 0
    loss_before = []
    step = 0
    scaler = torch.cuda.amp.GradScaler() if amp else None

    pbar = tqdm.tqdm(train_loader)
    for batch_data in pbar:
        pbar.set_description(f'Epoch:{epoch}')
        step += 1
        inputs, labels, masks, features = (
            batch_data[IMAGE_KEY].to(device),
            batch_data[LABEL_KEY].to(device),
            batch_data[MASK_KEY].to(device).float(),
            torch.as_tensor(batch_data[FEATURE_KEY]).to(device))
        optimizer.zero_grad()
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
            #! 暂时判断是resnet 只传入图像
                if 'res' in net:
                    outputs = model(inputs)
                else:
                    outputs = model(inputs, labels, features, masks)
                if loss_name == 'multi-BCE':
                    loss, label_loss, feature_loss = loss_function(outputs, labels, features)
                if loss_name == 'multi-CE':
                    loss, label_loss, feature_loss = loss_function(outputs, labels, features)
                elif loss_name == 'BCE':
                    if len(outputs.shape) == 1:
                        outputs = outputs
                    else:
                        outputs = outputs.squeeze(-1)
                    loss = loss_function(outputs, labels.to(torch.float32))
                elif loss_name == 'DWA-BCE':
                    loss, label_loss, feature_loss = loss_function(outputs, labels, features)
                elif loss_name == 'CE':
                    if len(outputs.shape) == 1:
                        outputs = outputs
                    else:
                        outputs = outputs.squeeze(-1)
                    loss = loss_function(outputs, labels.to(torch.long))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if 'res' in net:
                outputs = model(inputs)
            else:
                outputs = model(inputs, labels, features, masks)
            if loss_name == 'multi-BCE':
                loss, label_loss, feature_loss = loss_function(outputs, labels, features)
            if loss_name == 'multi-CE':
                loss, label_loss, feature_loss = loss_function(outputs, labels, features)
            elif loss_name == 'BCE':
                if len(outputs.shape) == 1:
                        outputs = outputs
                else:
                    outputs = outputs.squeeze()
                loss = loss_function(outputs, labels.to(torch.float32))
            elif loss_name == 'DWA-BCE':
                loss, label_loss, feature_loss = loss_function(outputs, labels, features)
            elif loss_name == 'CE':
                if len(outputs.shape) == 1:
                    outputs = outputs
                else:
                    outputs = outputs.squeeze(-1)
                loss = loss_function(outputs, labels.to(torch.long))
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()

        if 'DWA' in loss_name:
            epoch_loss_MB += label_loss
            epoch_loss_feature += feature_loss
            if len(outputs[1].shape) == 1:  ## batch size 为 1 的情况
                task_num = outputs[1].shape[0] + 1
            else:
                task_num = outputs[1].shape[1] + 1
            weight = [1] * task_num ## 前两代初始化权重

            if len(loss_before) == 2:
                #! relative descending rate
                r_l = [loss_before[-1][i]/loss_before[-2][i] for i in range(task_num)]
                r_l_sum = sum([math.exp(r_l[i]/0.7) for i in range(task_num)])
                weight = [(task_num*math.exp(r_l[i]/0.7))/r_l_sum for i in range(task_num)]
                loss_before[0] = loss_before[1]
            
            if len(outputs[1].shape) == 1:
                task_loss = [torch.mean(epoch_loss_feature[n]) for n in range(task_num-1)]
                task_loss_weight = [torch.mean(epoch_loss_feature[n])*weight[n] for n in range(task_num-1)]
            else:
                task_loss = [torch.mean(epoch_loss_feature[:, n]) for n in range(task_num-1)]
                task_loss_weight = [torch.mean(epoch_loss_feature[:, n])*weight[n] for n in range(task_num-1)]

            L_now = sum(task_loss_weight) + epoch_loss_MB*weight[-1]

            all_task_loss = task_loss + [epoch_loss_MB]
            if len(loss_before) < 2: 
                loss_before.append(all_task_loss) 
            else:
                loss_before[1] = all_task_loss

        epoch_len = len(train_ds)// train_loader.batch_size
        pbar.set_postfix({'loss' : '{0:.3f}'.format(loss.item())})
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    train_iters.append(epoch)


    if (epoch + 1) % valid_interval == 0:
        model.eval()
        with torch.no_grad():
            if amp:
                with torch.cuda.amp.autocast():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    y_feat_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y_feat = torch.tensor([], dtype=torch.long, device=device)
                    val_loss = 0
                    valid_step = 0
                    for val_data in valid_loader:
                        valid_step += 1
                        val_images, valid_labels, val_masks, val_features = (
                            val_data[IMAGE_KEY].to(device),
                            val_data[LABEL_KEY].to(device),
                            val_data[MASK_KEY].to(device).float(),
                            torch.as_tensor(val_data[FEATURE_KEY]).to(device),
                        )
                        # valid_labels =valid_labels if out_channels == 1 else AsDiscrete(to_onehot=True, n_classes=2)(valid_labels)
                        if 'res' in net:
                            val_outputs = model(val_images)
                        else:
                            val_outputs = model(val_images, valid_labels, val_features, val_masks)
                        y = torch.cat([y, valid_labels], dim=0)
                        y_feat = torch.cat([y_feat, val_features], dim=0)
                        if type(val_outputs) is tuple:
                            val_output1 = val_outputs[0]
                            val_output2 = val_outputs[1]

                            if len(val_output2.shape) == 1:
                                val_output2 = val_output2.unsqueeze(0)
                            if len(val_output1.shape) == 1:
                                val_output1 = val_output1.unsqueeze(0)
                            y_feat_pred = torch.cat([y_feat_pred, val_output2], dim=0)
                            y_pred = torch.cat([y_pred, val_output1], dim=0)
                        else:
                            if len(val_outputs.shape) == 1:
                                val_outputs = val_outputs.unsqueeze(0)
                            y_pred = torch.cat([y_pred, val_outputs], dim=0)
                
                        ##! 计算validation loss
                        if loss_name == 'multi-BCE':
                            val_loss_iter, _, _ = loss_function(val_outputs, valid_labels, val_features)
                        elif loss_name == 'multi-CE':
                            val_loss_iter, _, _ = loss_function(val_outputs, valid_labels, val_features)
                        elif loss_name == 'BCE':
                            if len(val_outputs.shape) == 1:
                                val_outputs = val_outputs
                            else:
                                val_outputs = val_outputs.squeeze(-1)
                            val_loss_iter = loss_function(val_outputs, valid_labels.to(torch.float32))
                        elif loss_name == 'DWA-BCE':
                            val_loss_iter, valid_loss_before = loss_function(valid_loss_before, val_outputs, valid_labels, val_features)
                        elif loss_name == 'CE':
                            if len(val_outputs.shape) == 1:
                                val_outputs = val_outputs
                            else:
                                val_outputs = val_outputs.squeeze(-1)
                            val_loss_iter = loss_function(val_outputs, valid_labels.to(torch.long))

                        val_loss += val_loss_iter
            else:
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                y_feat_pred = torch.tensor([], dtype=torch.float32, device=device)
                y_feat = torch.tensor([], dtype=torch.long, device=device)
                val_loss = 0
                valid_step = 0
                for val_data in valid_loader:
                    valid_step += 1
                    val_images, valid_labels, val_masks, val_features = (
                        val_data[IMAGE_KEY].to(device),
                        val_data[LABEL_KEY].to(device),
                        val_data[MASK_KEY].to(device).float(),
                        torch.as_tensor(val_data[FEATURE_KEY]).to(device),
                    )
                    # valid_labels =valid_labels if out_channels == 1 else AsDiscrete(to_onehot=True, n_classes=2)(valid_labels)
                    if 'res' in net:
                        val_outputs = model(val_images)
                    else:
                        val_outputs = model(val_images, valid_labels, val_features, val_masks)
                    y = torch.cat([y, valid_labels], dim=0)
                    y_feat = torch.cat([y_feat, val_features], dim=0)
                    if type(val_outputs) is tuple:
                        val_output1 = val_outputs[0]
                        val_output2 = val_outputs[1]
                        if len(val_output2.shape) == 1:
                            val_output2 = val_output2.unsqueeze(0)
                        if len(val_output1.shape) == 1:
                            val_output1 = val_output1.unsqueeze(0)
                        y_feat_pred = torch.cat([y_feat_pred, val_output2], dim=0)
                        y_pred = torch.cat([y_pred, val_output1], dim=0)
                    else:
                        y_pred = torch.cat([y_pred, val_outputs], dim=0)
            
                    ##! 计算validation loss
                    if loss_name == 'multi-BCE':
                        val_loss_iter, _, _ = loss_function(val_outputs, valid_labels, val_features)
                    elif loss_name == 'multi-CE':
                        val_loss_iter, _, _ = loss_function(val_outputs, valid_labels, val_features)
                    elif loss_name == 'BCE':
                        val_loss_iter = loss_function(val_outputs.squeeze(), valid_labels.to(torch.float32))
                    elif loss_name == 'DWA-BCE':
                        val_loss_iter, valid_loss_before = loss_function(valid_loss_before, val_outputs, valid_labels, val_features)
                    val_loss += val_loss_iter
            
            valid_iters.append(epoch)
            val_loss /= valid_step
            valid_loss_values.append(val_loss)

            if type(val_outputs) is tuple:
                if 'BCE' in loss_name:
                    y_feat_pred_act = Activations(sigmoid=True)(y_feat_pred)
                    feat_auc = []
                    for i in range(y_feat_pred_act.shape[1]):
                        _feat_auc = compute_roc_auc(y_feat_pred_act[:, i], y_feat[:, i])
                        feat_auc.append(_feat_auc)
                elif 'CE' in loss_name:
                    y_feat_pred_detach = torch.chunk(y_feat_pred, y_feat.shape[1], 1)
                    y_feat_detach = torch.chunk(y_feat, y_feat.shape[1], 1)
                    y_feat_pred_act = [Activations(softmax=True)(y_feat_pred_detach[i]) for i in range(y_feat.shape[1])]
                    y_feat = [AsDiscrete(to_onehot=True, n_classes=2)(a) for a in y_feat_detach]
                    feat_auc = []
                    for i in range(len(y_feat_pred_act)):
                        _feat_auc = compute_roc_auc(y_feat_pred_act[i], y_feat[i])
                        feat_auc.append(_feat_auc)
                epoch_feat_auc.append(feat_auc)
                del y_feat_pred_act
            else:
                feat_auc = [0, 0] ##只是为了计算，没有意义
            
            if out_channels == 1 :
                y_pred_act = Activations(sigmoid=True)(y_pred) 
            else :
                y_pred_act = Activations(softmax=True)(y_pred)
                y = AsDiscrete(to_onehot=True, n_classes=2)(y)
            auc_metric = compute_roc_auc(y_pred_act, y)
            metric_values.append(auc_metric)
            # all_auc = auc_metric + sum(feat_auc)
            all_auc = 5*auc_metric + 2*sum(feat_auc)
            # all_auc = 5*auc_metric + sum(feat_auc[:-1]) + 5*feat_auc[-1]
            acc_value = torch.eq(AsDiscrete(threshold_values=True, logit_thresh=0.5)(y_pred_act).squeeze(), y)
            del y_pred_act
            acc_metric = acc_value.sum().item() / len(acc_value)
            model_saver_SBA(auc_metric, epoch)
            model_saver_SBL(-val_loss, epoch)
            model_saver_SBAA(all_auc, epoch)
            lr_scheduler.step(auc_metric)
            learning_rate.append(lr_scheduler._last_lr[0])

            early_stopping(val_loss, model)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(
                f"current epoch: {epoch + 1} current AUC: {auc_metric:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" current lr: {lr_scheduler._last_lr[0]:.4f}"
            )
        train_plt = dict({'loss': epoch_loss_values})
        valid_plt = {'auc': metric_values}
        if type(val_outputs) is tuple:
            epoch_feat_auc_arr = np.array(epoch_feat_auc)
            _valid_plt = {f'feat{i}_auc': epoch_feat_auc_arr[:,i] for i in range(epoch_feat_auc_arr.shape[1])}
            valid_plt.update(_valid_plt)
        
        plot_scores(
            epoch,
            n_epoch,
            train_iters,
            valid_iters,
            train_plt,
            valid_plt,
            len(train_ds),
            len(valid_ds),
            os.path.join(out_dir,'results.png'),
        )
        plot_fig(
            epoch,
            n_epoch,
            valid_iters,
            {'lr': learning_rate},
            len(train_ds),
            os.path.join(out_dir,'lr.png'),
        )
        
