import numpy as np
import os
import imageio
import math
import torch


# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes

    return image_numpy.astype(imtype)


# conver a tensor into a numpy array
def tensor2array(value_tensor):
    if value_tensor.dim() == 3:
        numpy = value_tensor.view(-1).cpu().float().numpy()
    else:
        numpy = value_tensor[0].view(-1).cpu().float().numpy()
    return numpy


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_network(net,path):
    try:
        net.load_state_dict(torch.load(path))
    except:
        pretrained_dict = torch.load(path)
        model_dict = net.state_dict()
        try:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            net.load_state_dict(pretrained_dict)
            print('Pretrained network has excessive layers; Only loading layers that are used')
        except:
            print('Pretrained network  has fewer layers; The following are not initialized:')
            not_initialized = set()
            for k, v in pretrained_dict.items():
                if v.size() == model_dict[k].size():
                    model_dict[k] = v

            for k, v in model_dict.items():
                if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                    not_initialized.add(k.split('.')[0])
            print(sorted(not_initialized))
            net.load_state_dict(model_dict)

    return net
