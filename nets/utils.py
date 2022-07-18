import tqdm, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.cm as mpl_color_map
from PIL import Image
from pathlib import Path
from monai.utils import InterpolateMode
from monai.transforms import ScaleIntensity
from monai.transforms.post.array import Activations

def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for layer in c:
            apply_leaf(layer, f)


def set_trainable(module, b):
    apply_leaf(module, lambda m: set_trainable_attr(m, b))


def default_normalizer(acti_map) -> np.ndarray:
    """
    A linear intensity scaling by mapping the (min, max) to (1, 0).
    """
    if isinstance(acti_map, torch.Tensor):
        acti_map = acti_map.detach().cpu().numpy()
    scaler = ScaleIntensity(minv=1.0, maxv=0.0)
    acti_map = [scaler(x) for x in acti_map]
    return np.stack(acti_map, axis=0)

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


def save_activation(inputs, outputs, coeffi, label, prediction, out_size, class_idx, save_path):
    '''
    inputs: BCWH(D)
    outputs: B*C*spatial_size
    weight: Channel*Class
    label: true label
    prediction: pred result
    spatial_size: activation_map size
    out_size: output save size
    class_idx: class choose to show the attention map
    '''
    label = label.unsqueeze(-1) if len(label.shape) < 2 else label
    out_act = Activations(sigmoid=True)(prediction)
    if len(inputs.shape) == 5:
        dim = 3 
        batch_sz, channel_sz, _, _, _ = inputs.shape
    else:
        dim = 2
        batch_sz, channel_sz, _, _ = inputs.shape
    linear_mode = [InterpolateMode.LINEAR, InterpolateMode.BILINEAR, InterpolateMode.TRILINEAR]
    interp_mode = linear_mode[dim - 1]
    for i in tqdm.tqdm(range(batch_sz)):
        activation_map = torch.zeros(outputs.shape[2:]).to('cuda')
        for c in range(coeffi.shape[1]):
            _activation_map = outputs[i,c, ...]*coeffi[class_idx, c]
            activation_map = torch.add(activation_map, _activation_map)
        activation_map_act = F.relu(activation_map)
        # activation_arr = activation_map.detach().cpu().numpy()
        # activation_arr_norm = np.uint8(((activation_arr-np.min(activation_arr))/(np.max(activation_arr)-np.min(activation_arr)))*255)
        # act_map = np.uint8(Image.fromarray(activation_arr_norm).resize(out_size, Image.ANTIALIAS))/255
        
        if dim == 2: 
            origin_img = inputs.cpu().detach().numpy().squeeze()[i,...]
            activation_arr_up = F.interpolate(torch.tensor(activation_map_act).unsqueeze(0).unsqueeze(0), size=inputs.shape[2:], mode=str(interp_mode.value), align_corners=False)
            activation_arr_up_norm = default_normalizer(activation_arr_up)
            nib.save(
                        nib.Nifti1Image(activation_arr_up_norm.squeeze(0), np.eye(4)), Path(save_path)/f"{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}-act.nii.gz"
                    )
            nib.save(
                        nib.Nifti1Image(inputs.cpu().detach().numpy().squeeze()[i,...], np.eye(4)), Path(save_path)/f"{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}-img.nii.gz"
                    )
        elif channel_sz == 1 and dim == 3: 
            origin_img = inputs.cpu().detach().numpy().squeeze()[i,...,int((inputs.shape[-1]+1)/2)]
            activation_arr_up = F.interpolate(torch.tensor(activation_map_act).unsqueeze(0).unsqueeze(0), size=inputs.shape[2:], mode=str(interp_mode.value), align_corners=False)
            activation_arr_up_norm = default_normalizer(activation_arr_up)
            if len(out_act.shape) == 1:
                nib.save(
                            nib.Nifti1Image(activation_arr_up_norm.squeeze(), np.eye(4)), Path(save_path)/f"{i}-GT-{label.squeeze()[class_idx]}-{out_act[class_idx]:.3f}-act.nii.gz"
                        )
                nib.save(
                            nib.Nifti1Image(inputs.cpu().detach().numpy().squeeze(), np.eye(4)), Path(save_path)/f"{i}-GT-{label.squeeze()[class_idx]}-{out_act[class_idx]:.3f}-img.nii.gz"
                        )
            else:
                nib.save(
                            nib.Nifti1Image(activation_arr_up_norm.squeeze(), np.eye(4)), Path(save_path)/f"{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}-act.nii.gz"
                        )
                nib.save(
                            nib.Nifti1Image(inputs.cpu().detach().numpy().squeeze()[i,...], np.eye(4)), Path(save_path)/f"{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}-img.nii.gz"
                        )
        else:
            origin_img = inputs.cpu().detach().numpy().squeeze()[i,int((channel_sz+1)/2),...]
