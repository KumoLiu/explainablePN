from torch._C import dtype
from torchvision.models import resnet101
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datasets import CLASSIFICATION_DATASETS
from monai.data import DataLoader
from monai.visualize import GradCAM, CAM
from monai.transforms import AsDiscrete, Activations
from utils_cw import check_dir, Print
from utils import apply_colormap_on_image

IMAGE_KEY = 'image'
LABEL_KEY = 'label'
MASK_KEY = 'mask'

def vis_gradcam(
    params,
    test_list,
    model_path,
    out_dir,
    target_layer,
    **kwargs,
):
    dataset_name = params['dataset_name']
    archi_name = params['archi_name']
    mode = params['mode']
    input_nc = params['input_nc']
    output_nc = params['output_nc']
    resize_shape = kwargs.get('resize_shape', None)
    device = kwargs.get('device', torch.device("cuda:0"))
    verbose = kwargs.get('verbose')
    visiualize = kwargs.get('visiualize')
    
    dataset_type = CLASSIFICATION_DATASETS['2D'][dataset_name]
    test_ds = dataset_type(test_list, 'test', {'input_nc':input_nc})
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=5)

    # load model
    model = get_network(archi_name,
                        mode=mode,
                        in_channels=input_nc,
                        out_channels=output_nc,
                        device=device,
                        spatial_size=(64,64))
    print(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    layer_name = []
    for name, _ in model.named_modules():
        layer_name.append(name)
    print('layer name ==>')
    print(layer_name)
    
    # initialize gradcam
    if visiualize == 'gradcam':
        cam = GradCAM(nn_module=model, target_layers=target_layer)    
    elif visiualize == 'cam':
        cam = CAM(nn_module=model, target_layers=target_layer)    

    print('raw_cam_size:', cam.feature_map_size((1,1,64,64), device=device))
    # # print(model)
    save_dir = check_dir(out_dir/visiualize)
    for _, test_data in enumerate(test_loader):
        test_images, test_labels, test_masks = (
                test_data[IMAGE_KEY].to(device),
                test_data[LABEL_KEY].to(device),
                test_data[MASK_KEY].to(device),
            )
        save_name = Path(test_data['image_meta_dict']['filename_or_obj'][0]).parent.name
        
        outputs_ = model(test_images)
        if  type(outputs_) is tuple:
            y_pred = outputs_[1]
        else:
            y_pred = outputs_
        if output_nc > 1:
            y_pred_act = Activations(softmax=True)(y_pred)
        else:
            y_pred_act = Activations(sigmoid=True)(y_pred)
        y_pred_act = y_pred_act.squeeze()

        #! get layer name ==> for name, _ in model.named_modules(): print(name) #  unet'down_4.convs.conv_1'; final_conv.conv
        Print('test_image', test_images.shape, verbose=verbose)
        # cam_result = cam(test_images, class_idx=1).squeeze()
        cam_result = cam(test_images).squeeze()
        Print('cam_result', cam_result.shape, verbose=verbose)

        # normalize origin img
        origin_img = test_images.cpu().detach().numpy().squeeze()
        origin_img = np.uint8(((origin_img - np.min(origin_img)) / (np.max(origin_img) - np.min(origin_img)))*255)
        origin_img = Image.fromarray(origin_img)

        test_masks = test_masks.cpu().detach().numpy().squeeze()
        test_masks = np.uint8(test_masks*255)
        test_masks = Image.fromarray(test_masks)

        if resize_shape is not None:
            origin_img = origin_img.resize(resize_shape,resample=Image.ANTIALIAS)
            cam_result = np.uint8(cam_result * 255)
            cam_result = np.uint8(Image.fromarray(cam_result).resize(resize_shape, Image.ANTIALIAS))/255
        no_trans_heatmap, heatmap_on_image = apply_colormap_on_image(origin_img, cam_result, 'jet')
        
        # origin_img.save(save_dir/f'{i}_origin.png')
        heatmap_on_image.save(save_dir/f'{save_name}_heatmap_on_img_label-{test_labels.item()}_pred-{y_pred_act[1].item():0.3f}.png')
        test_masks.save(save_dir/f'{save_name}_mask_{test_labels.item()}.png')
    

#! test gradcam
# image = Path("/homes/yliu/cat_dog.png")
# img_arr = np.array(Image.open(image).convert('RGB'))
# img_arr = img_arr.transpose((2,0,1))
# img_arr_N = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
# img_N_tensor = torch.tensor(img_arr_N, dtype=torch.float32).unsqueeze(0)

# model = resnet101(pretrained=True)
# cam = GradCAM(nn_module=model, target_layers='layer4')
# cam_result = np.array(cam(img_N_tensor, class_idx=243).squeeze())
# print('cam_result', cam_result.shape)
# no_trans_heatmap, heatmap_on_image = apply_colormap_on_image(Image.open(image).convert('RGB'), cam_result, 'hsv')
# heatmap_on_image.save('/homes/yliu/cam.png')