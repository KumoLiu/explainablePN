import torch
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as mpl_color_map
from ..utils import check_dir
import torch.nn.functional as F

def transfer_text_to_img(text, image_size, font_size, xy):
    # xy: 文字开始的位置
    img = Image.new('RGB', image_size, (255, 255, 255))
    dr = ImageDraw.Draw(img)
    font = ImageFont.truetype(os.path.join("/homes/yliu", "STZHONGS.TTF"), font_size)
    dr.text(xy, text, font=font, fill="#000000")
    return img


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


def save_activation_mb_npz(file_names, outputs, coeffi, label, pred, save_path):
    batch_sz, channel_sz = outputs.shape[0], outputs.shape[1]
    class_sz, coeffi_channel_sz = coeffi.shape[0], coeffi.shape[1]
    pred = F.sigmoid(pred).detach().cpu().numpy()
    for idx in range(batch_sz):
        activation_map = torch.zeros(outputs.shape[2:]).to('cuda')
        for c in range(coeffi_channel_sz):
            _activation_map = outputs[idx,c, ...] * coeffi[0, c]
            activation_map = torch.add(activation_map, _activation_map)
        activation_arr = activation_map.detach().cpu().numpy()
        save_target = check_dir(save_path/file_names[idx]/f'mb_{label[idx]}_{pred[idx, 0]:.2f}.npz', isFile=True)
        np.savez(save_target, feature_arr=activation_arr)


def save_activation_feat_npz(file_names, outputs, coeffi, label, pred, save_path):
    """
    Only save activation map as numpy array.
    """
    batch_sz, channel_sz = outputs.shape[0], outputs.shape[1]
    class_sz, coeffi_channel_sz = coeffi.shape[0], coeffi.shape[1]
    pred = F.sigmoid(pred).detach().cpu().numpy()
    for idx in range(batch_sz):
        ## 保存所有的征象的activation map
        activation_map_list = []
        for _class in range(class_sz):
            activation_map = torch.zeros(outputs.shape[2:]).to('cuda')
            for _channel in range(coeffi_channel_sz):
                _activation_map = outputs[idx,_channel, ...] * coeffi[_class, _channel]
                activation_map = torch.add(activation_map, _activation_map)
            activation_arr = activation_map.detach().cpu().numpy()
            save_target = check_dir(save_path/file_names[idx]/f'feature_{_class}_{pred[idx, _class]:.2f}.npz', isFile=True)

            np.savez(save_target, feature_arr=activation_arr)

            # activation_arr_norm = np.uint8(((activation_arr-np.min(activation_arr))/(np.max(activation_arr)-np.min(activation_arr)))*255)
            # act_map = np.uint8(Image.fromarray(activation_arr_norm).resize(out_size, Image.ANTIALIAS))/255
            # activation_map_list.append(act_map)



def save_result(file_names, inputs, outputs, coeffi, label, prediction, out_size, class_idx, save_path):
    '''
    inputs: BCWH(D)
    outputs: feature map from final conv, B*C*spatial_size
    coeffi: weight from final fc layer, Class*Channel
    label: true label
    prediction: pred result
    spatial_size: activation_map size
    out_size: output save size
    class_idx: class choose to show the attention map
    '''
    label = label.unsqueeze(-1) if len(label.shape) < 2 else label
    out_act = Activations(sigmoid=True)(prediction)
    batch_sz = inputs.shape[0]
    channel_sz = inputs.shape[1]
    class_sz = coeffi.shape[0]
    coeffi_channel_sz = coeffi.shape[1]
    for i in tqdm.tqdm(range(batch_sz)):
        ## 保存所有的征象的activation map
        activation_map_list = []
        for _class in range(class_sz):
            activation_map = torch.zeros(outputs.shape[2:]).to('cuda')
            for _channel in range(coeffi_channel_sz):
                _activation_map = outputs[i,_channel, ...]*coeffi[_class, _channel]
                activation_map = torch.add(activation_map, _activation_map)
            activation_arr = activation_map.detach().cpu().numpy()
            activation_arr_norm = np.uint8(((activation_arr-np.min(activation_arr))/(np.max(activation_arr)-np.min(activation_arr)))*255)
            act_map = np.uint8(Image.fromarray(activation_arr_norm).resize(out_size, Image.ANTIALIAS))/255
            activation_map_list.append(act_map)

        # 生成中间层原始图像
        if channel_sz == 1: 
            origin_img = inputs.cpu().detach().numpy().squeeze()[i,...]
        else:
            origin_img = inputs.cpu().detach().numpy().squeeze()[i,int((channel_sz+1)/2),...]
        origin_img = np.uint8(((origin_img - np.min(origin_img)) / (np.max(origin_img) - np.min(origin_img)))*255)
        origin_img = Image.fromarray(origin_img)
        origin_img = origin_img.resize(out_size, resample=Image.ANTIALIAS)

        
        no_trans_heatmap_list, heatmap_on_image_list= [], []
        for _class in range(class_sz):
            no_trans_heatmap, heatmap_on_image = apply_colormap_on_image(origin_img, activation_map_list[_class], 'jet')

        ## 保存第一次读片的图
        first_img_size = (out_size[0]*5+100, out_size[1]+50)
        new_origin_img = Image.new('RGB', first_img_size, 255)
        text1 = transfer_text_to_img('第一次读片', (out_size[0]*5+100,50), 30, ((out_size[0]*5+100)//2, 5))
        text2 = transfer_text_to_img('CT图', (100,out_size[1]), 30, (25, (out_size[1]-40)//2))
        inputs_arr = inputs.cpu().detach().numpy().squeeze()
        x = y = 0
        new_origin_img.paste(text1, (x, y))  
        y += 50
        new_origin_img.paste(text2, (x, y))  
        x += 100
        for _channel_sz in range(channel_sz):
            origin_img_slice = inputs_arr[i, _channel_sz, ...]
            origin_img_slice = np.uint8(((origin_img_slice - np.min(origin_img_slice)) / (np.max(origin_img_slice) - np.min(origin_img_slice)))*255)
            origin_img_slice = Image.fromarray(origin_img_slice)
            origin_img_slice = origin_img_slice.resize(out_size, resample=Image.ANTIALIAS)
            new_origin_img.paste(origin_img_slice, (x, y))  
            x += out_size[1]
        
        # new_origin_img.save(Path(save_path)/f"origin_img-{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}.png")
        new_origin_img.save(Path(save_path)/f"origin_img-{i}-{out_act[i, class_idx]:.3f}.png")


        ## 保存第二次读片的图
        second_img_size = (out_size[0]*4+100, out_size[1]+50)
        text3 = transfer_text_to_img('AI诊断', (100,out_size[1]), 30, (25, (out_size[1]-40)//2))
        text4 = transfer_text_to_img(f'恶性概率   易发现度  钙化   纹理   球状度   边缘\n{out_act[i, class_idx]:.3f}', (100,out_size[1]), 30, (25, (out_size[1]-40)//2))
        new_img = Image.new('RGB', second_img_size, 255)
        x = y = 0
        new_img.paste(text4, (x, y))
        y += 50
        new_img.paste(text3, (x, y))
        x += 30
        new_img.paste(origin_img, (x, y))  
        x += out_size[1] 
        new_img.paste(no_trans_heatmap, (x, y))  
        x += out_size[1]
        new_img.paste(heatmap_on_image, (x, y))  
        x += out_size[1] 
        new_img.paste(attention_map, (x, y))  
        x += out_size[1] 

        new_img.save(Path(save_path)/f"{class_idx}-{i}-GT-{label[i, class_idx]}-{out_act[i, class_idx]:.3f}.png")
