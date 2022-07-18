import os
from pathlib import Path
import math
import tqdm
import copy
import torch
import yaml
import json
import seaborn as sns
import numpy as np
from PIL import Image
from monai.transforms.post.array import Activations
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.cm as mpl_color_map
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from nets.hesam_aag import hesam_aag, hesam_woaag, raw_hesam_aag
from nets.resnet import resnet18, resnet50, resnet50_sam
from nets.resnet_agg_sam import resnet18_aag_sam, resnet34_aag_sam, resnet34_aag_sam_v2
from nets.HESAM import HESAM, HESAM2, HESAM_CABM, HESAM_CABM_2head
from nets.unet_cls import DynUNet_cls

color_list = sns.color_palette('deep') + sns.color_palette('bright')
def DrawROCList(pred_list, label_list, name_list='', store_path='', is_show=True, fig=plt.figure()):
    '''
    To Draw the ROC curve.
    :param pred_list: The list of the prediction.
    :param label_list: The list of the label.
    :param name_list: The list of the legend name.
    :param store_path: The store path. Support png and eps.
    :return: None

    '''
    if not isinstance(pred_list, list):
        pred_list = [pred_list]
    if not isinstance(label_list, list):
        label_list = [label_list]
    if not isinstance(name_list, list):
        name_list = [name_list]

    fig.clear()
    axes = fig.add_subplot(1, 1, 1)

    for index in range(len(pred_list)):
        fpr, tpr, threshold = roc_curve(label_list[index], pred_list[index])
        auc = roc_auc_score(label_list[index], pred_list[index])
        name_list[index] = name_list[index] + (' (AUC = %0.3f)' % auc)

        axes.plot(fpr, tpr, color=color_list[index], label='ROC curve (AUC = %0.3f)' % auc,linewidth=3)

    axes.plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.05)
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_title('Receiver operating characteristic curve')
    axes.legend(name_list, loc="lower right")
    if store_path:
        fig.set_tight_layout(True)
        if store_path[-3:] == 'png':
            fig.savefig(store_path, dpi=300, format='png')
        elif store_path[-3:] == 'eps':
            fig.savefig(store_path, dpi=1200, format='eps')

    if is_show:
        plt.show()

    return axes


COLORS = ['g','r','c','m','y','k','b']
def plot_fig(
    epoch,
    total_epoch,
    train_iter,
    scores,
    train_data_num,
    output_fpath,
):
    try:
        # plot testing and training score
        f = plt.figure(1)
        plt.clf()
        for i, (k, v) in enumerate(scores.items()):
            plt.plot(train_iter, v, label='{}'.format(k), color=COLORS[i], linewidth=2.0)
        plt.title('Training: iter {} of {}'.format(epoch, total_epoch))
        plt.ylim([0.,0.5])
        ax = plt.axes()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.ylabel('learning rate')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)
        f.show()

        f.savefig(output_fpath)
    except BaseException as e:
        print('Failed to do plot: ' + str(e))

class Save_best_n_models(object):
    def __init__(self, model, out_dir, n_saved=3):
        self.model = model
        self.n_saved = n_saved
        self._saved = []
        self.best_metric = -np.inf
        self.out_dir = out_dir
    
    def _check_lt_n_saved(self, or_equal=False):
        if self.n_saved is None:
            return True

        return len(self._saved) < self.n_saved + int(or_equal)

    def __call__(self, metric, epoch):
        fname = os.path.join(self.out_dir, f'BestModel@{epoch}with{metric:.3f}.pt')
        if self._check_lt_n_saved():
            torch.save(self.model.state_dict(), fname)
            self._saved.append({'metric':metric, 'fname':fname})
            self._saved.sort(key=lambda item: item['metric'])
        elif self._saved[0]['metric'] < metric:
            torch.save(self.model.state_dict(), fname)
            item = self._saved.pop(0)
            if os.path.isfile(item['fname']):
                os.unlink(item['fname'])
            self._saved.append({'metric':metric, 'fname':fname})
            self._saved.sort(key=lambda item: item['metric'])
        else:
            pass


def get_network(name,
                dimensions, 
                in_channels,
                out_channels,
                out_channels_f,
                features,
                device,
                mode, 
                save_attentionmap_fpath=False,
                use_attention=False,
                use_cbam=False,
                use_mask=False,
                use_aspp=False,
                save_latent=False
                ):
    if name == 'res50':
        model = resnet50(
            pretrained=False,
            in_channels=in_channels,
            num_classes=out_channels,
            mode=mode,
        ).to(device)
    elif name == 'res50-sam':
        model = resnet50_sam(
            pretrained=False,
            in_channels=in_channels,
            num_classes=out_channels,
            mode=mode,
        ).to(device)
    elif name == 'hesam':
        model = HESAM(
            dimensions=dimensions,
            features=features,
            in_channels=in_channels,
            out_channels=out_channels,
            use_attention = use_attention,
            use_cbam = use_cbam,
            use_mask = use_mask,
            save_attentionmap_fpath=save_attentionmap_fpath,
            save_latent=save_latent
        ).to(device)
    elif name == 'hesam2':
        model = HESAM2(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            mode=mode,
            save_attentionmap_fpath=save_attentionmap_fpath,
        ).to(device)
    elif name == 'hesam_cbam':
        model = HESAM_CABM(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            mode=mode,
            use_attention = use_attention,
            use_cbam = use_cbam,
            use_mask = use_mask,
            save_attentionmap_fpath=save_attentionmap_fpath,
        ).to(device)
    elif name == 'hesam_cbam-2head':
        model = HESAM_CABM_2head(
            dimensions=dimensions,
            features=features,
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_f=out_channels_f,
            dropout=0.0,
            mode=mode,
            use_attention = use_attention,
            use_cbam = use_cbam,
            use_mask = use_mask,
            use_aspp = use_aspp,
            save_attentionmap_fpath=save_attentionmap_fpath,
            save_latent=save_latent,
        ).to(device)
    elif name == 'Resnet34_agg_sam':
        model = resnet34_aag_sam(
            pretrained_model_path=None,
            dim=dimensions,
            in_channels=in_channels,
            roi_classes=3,
            num_classes=out_channels,
            out_channels_f=out_channels_f,
            sam_size=6,
            use_cbam=use_cbam,
            use_mask=use_mask,
        ).to(device)
    elif name == 'Resnet34_agg_sam_multiout':
        model = resnet34_aag_sam_v2(
            pretrained_model_path=None,
            dim=dimensions,
            in_channels=in_channels,
            roi_classes=3,
            num_classes=out_channels,
            out_channels_f=out_channels_f,
            sam_size=6,
            use_cbam=use_cbam,
            use_mask=use_mask,
        ).to(device)
    elif name == 'hesam_agg':
        model = hesam_aag(
            dimensions=dimensions,
            features=features,
            in_channels=in_channels,
            roi_classes=1,
            out_channels=out_channels,
            out_channels_f=out_channels_f,
            sam_size=6,
            use_cbam=use_cbam,
            use_mask=use_mask,
            save_attentionmap_fpath=save_attentionmap_fpath,
        ).to(device)
    elif name == 'hesam_woagg':
        model = hesam_woaag(
            dimensions=dimensions,
            features=features,
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_f=out_channels_f,
            sam_size=6,
            use_cbam=use_cbam,
            use_mask=use_mask,
            save_attentionmap_fpath=save_attentionmap_fpath,
        ).to(device)
    elif name == 'raw_hesam_agg':
        model = raw_hesam_aag(
            dimensions=dimensions,
            features=features,
            in_channels=in_channels,
            roi_classes=1,
            out_channels=out_channels,
            out_channels_f=out_channels_f,
            sam_size=6,
            use_cbam=use_cbam,
            use_mask=use_mask,
            save_attentionmap_fpath=save_attentionmap_fpath,
            save_latent=save_latent,
        ).to(device)
    elif name == 'unet_cls':
        n_depth = 5 if n_depth == -1 else n_depth
        kernel_size = (3,) + (3,) * n_depth
        strides = (1,) + (2,) * n_depth
        upsample_kernel_size = (1,) + (2,) * n_depth
        res_block = False
        model = DynUNet_cls(
            spatial_dim=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            res_block=res_block,
        )
    else:
        raise NotImplementedError
    
    return model

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def binarization(feat_list):
    return list(map(lambda x: int(x>3), feat_list))

COLORS = ['g','r','c','m','y','k','b','wheat','chocolate', 'black']
def plot_scores(
    epoch,
    total_epoch,
    train_iter,
    valid_iter,
    train_scores,
    valid_scores,
    train_data_num,
    valid_data_num,
    output_fpath,
):
    try:
        # plot testing and training score
        f = plt.figure(1)
        plt.clf()
        for i, (k, v) in enumerate(train_scores.items()):
            plt.plot(train_iter, v, label='train {}'.format(k), color=COLORS[i], linewidth=2.0)
        for i, (k, v) in enumerate(valid_scores.items()):
            plt.plot(valid_iter, v, label='valid {}'.format(k), color=COLORS[i+1], linewidth=2.0)
        plt.title('Training: iter {} of {}'.format(epoch, total_epoch))
        plt.ylim([0.,1.])
        ax = plt.axes()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.ylabel('AUC')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)
        f.show()

        f.savefig(output_fpath)
    except BaseException as e:
        print('Failed to do plot: ' + str(e))

    

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_attention(inputs, outputs, coeffi, label, prediction, out_size, class_idx, save_path):
    '''
    inputs: BCWH(D)
    outputs: B*C*spatial_size
    weight: Channel*Class
    label: true label
    prediction: pred result
    spatial_size: attention_map size
    out_size: output save size
    class_idx: class choose to show the attention map
    '''
    label = label.unsqueeze(-1) if len(label.shape) < 2 else label
    out_act = Activations(sigmoid=True)(prediction)
    for i in tqdm.tqdm(range(inputs.shape[0])):
        attention_map = torch.zeros(outputs.shape[2:]).to('cuda')
        for c in range(coeffi.shape[1]):
            _attention_map = outputs[i,c, ...]*coeffi[class_idx, c]
            attention_map = torch.add(attention_map, _attention_map)
        attention_arr = attention_map.detach().cpu().numpy()
        attention_arr_norm = np.uint8(((attention_arr-np.min(attention_arr))/(np.max(attention_arr)-np.min(attention_arr)))*255)
        att_map = np.uint8(Image.fromarray(attention_arr_norm).resize(out_size, Image.ANTIALIAS))/255

        origin_img = inputs.cpu().detach().numpy().squeeze()[i,2,...]
        origin_img = np.uint8(((origin_img - np.min(origin_img)) / (np.max(origin_img) - np.min(origin_img)))*255)
        origin_img = Image.fromarray(origin_img)
        origin_img = origin_img.resize(out_size, resample=Image.ANTIALIAS)
        no_trans_heatmap, heatmap_on_image = apply_colormap_on_image(origin_img, att_map, 'jet')
        new_img = Image.new('RGB', (out_size[0]*3, out_size[1]), 255)
        x = y = 0
        new_img.paste(origin_img, (x, y))  
        x += out_size[1] 
        new_img.paste(no_trans_heatmap, (x, y))  
        x += out_size[1]
        new_img.paste(heatmap_on_image, (x, y))  
        x += out_size[1] 

        new_img.save(Path(save_path)/f"{class_idx}-{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}.png")
        # heatmap_on_image.save(Path(save_path)/f"FEAT{class_idx}-{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}.png")
        # no_trans_heatmap.save(Path(save_path)/f"HM{class_idx}-{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}.png")
        # origin_img.save(Path(save_path)/f"RAW{class_idx}-{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}.png")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



class cls_bce_loss(nn.Module):
    def __init__(self, pos_weight, weight, reduction='mean'):
        super(cls_bce_loss, self).__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.weight = weight
        #! lidc 4 jsph 1.75
        self.bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4).to('cuda'))
        self.bce2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to('cuda'), reduction='none')
    
    def forward(self, outputs, labels, features):
        output1 = outputs[0].squeeze()
        output2 = outputs[1]
        if len(output1.shape) < 1:
            output1 = output1.unsqueeze(0)
        if len(features.shape) < 1:
            features = features.unsqueeze(0)
        bce_loss = self.bce1(output1, labels.to(torch.float32))
        # bce_loss = 0.
        ce_loss = self.bce2(output2, features.squeeze().to(torch.float32))


        final_ce_loss = 0
        _len = 0
        for i in range(len(self.pos_weight)):
            if len(output2.shape) == 1:
                feat_loss = torch.mean(ce_loss[i])
            else:
                feat_loss = torch.mean(ce_loss[:, i])
            final_ce_loss += feat_loss * (1/self.weight[i])
            _len += 1/self.weight[i]
        final_ce_loss /= _len
        # final_ce_loss = (2*feat1_loss + feat2_loss + 3*feat3_loss + 4*feat4_loss)/10

        if self.reduction == "sum":
            result = bce_loss + final_ce_loss
        elif self.reduction == 'mean':
            result = (bce_loss + final_ce_loss)/2
        elif self.reduction == 'weight':
            result = 0.4*bce_loss + 0.6*final_ce_loss
        else:
            raise NotImplementedError
        return result, bce_loss, final_ce_loss

class DWA_bce_loss(nn.Module):
    def __init__(self, pos_weight):
        super(DWA_bce_loss, self).__init__()
        self.pos_weight = pos_weight
        self.bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.75).to('cuda'))
        self.bce2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to('cuda'), reduction='none')
    
    def forward(self, loss_before, outputs, labels, features):
        #! Loss_l 上一代loss
        #! Loss_ll 上上一代loss
        output1 = outputs[0].squeeze()
        output2 = outputs[1]
        bce_loss = self.bce1(output1, labels.to(torch.float32).squeeze())
        ce_loss = self.bce2(output2, features.to(torch.float32).squeeze())

        if len(output2.shape) == 1:  ## batch size 为 1 的情况
            task_num = output2.shape[0] + 1
        else:
            task_num = output2.shape[1] + 1
        weight = [1] * task_num

        if len(loss_before) == 2:
            #! relative descending rate
            r_l = [loss_before[-1][i]/loss_before[-2][i] for i in range(task_num)]
            r_l_sum = sum([math.exp(r_l[i]/0.7) for i in range(task_num)])
            weight = [(task_num*math.exp(r_l[i]/0.7))/r_l_sum for i in range(task_num)]
            loss_before[0] = loss_before[1]
        
        if len(output2.shape) == 1:
            task_loss = [torch.mean(ce_loss[n]) for n in range(task_num-1)]
            task_loss_weight = [torch.mean(ce_loss[n])*weight[n] for n in range(task_num-1)]
        else:
            task_loss = [torch.mean(ce_loss[:, n]) for n in range(task_num-1)]
            task_loss_weight = [torch.mean(ce_loss[:, n])*weight[n] for n in range(task_num-1)]

        L_now = sum(task_loss_weight) + bce_loss*weight[-1]

        all_task_loss = task_loss + [bce_loss]
        if len(loss_before) < 2: 
            loss_before.append(all_task_loss) 
        else:
            loss_before[1] = all_task_loss
    
        return L_now, loss_before
        
class cls_ce_loss(nn.Module):
    def __init__(self, pos_weight, weight, reduction='mean'):
        super(cls_ce_loss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        # self.bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]).to('cuda'))
        self.ce1 = nn.CrossEntropyLoss()
        self.ce2 = [nn.CrossEntropyLoss(weight=torch.tensor(pos_weight[i]).to('cuda'), reduction='mean') for i in range(len(pos_weight))]
    
    def forward(self, outputs, labels, features):
        feat_sum = features.shape[1]
        feat = torch.chunk(features, feat_sum, 1)
        output1 = outputs[0].squeeze()
        output2 = torch.chunk(outputs[1], feat_sum, 1)
        
        bce_loss = self.ce1(output1, labels)

        final_ce_loss = 0
        _len = 0
        for i in range(len(self.pos_weight)):
            feat_loss = self.ce2[i](output2[i], feat[i].squeeze())
            final_ce_loss += feat_loss * (1/self.weight[i])
            _len += 1/self.weight[i]
        final_ce_loss /= _len
        # final_ce_loss = (2*feat1_loss + feat2_loss + 3*feat3_loss + 4*feat4_loss)/10

        if self.reduction == "sum":
            result = bce_loss + final_ce_loss
        elif self.reduction == 'mean':
            result = (bce_loss + final_ce_loss)/2
        elif self.reduction == 'weight':
            result = 0.4*bce_loss + 0.6*final_ce_loss
        else:
            raise NotImplementedError
        return result, bce_loss, final_ce_loss


def check_dir(*arg, isFile=False, exist_ok=True):
    path = Path(*arg)
    if isFile:
        filename = path.name
        path = path.parent

    if not path.is_dir():
        os.makedirs(path, exist_ok=exist_ok)
    return path / filename if isFile else path


def get_items_from_file(filelist, format="auto", sep="\n"):
    """
    Simple wrapper for reading items from file.
    If file is dumped by yaml or json, set `format` to `json`/`yaml`.
    """
    filelist = Path(filelist)
    if not filelist.is_file():
        raise FileNotFoundError(f"No such file: {filelist}")

    if format == "auto":
        if filelist.suffix in [".json"]:
            format = "json"
        elif filelist.suffix in [".yaml", ".yml"]:
            format = "yaml"
        else:
            format = None

    with filelist.open() as f:
        if format == "yaml":
            lines = yaml.full_load(f)
        elif format == "json":
            lines = json.load(f)
        else:
            lines = f.read().split(sep)
    return lines


class PathlibEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return os.fspath(obj)
        return json.JSONEncoder.default(self, obj)

