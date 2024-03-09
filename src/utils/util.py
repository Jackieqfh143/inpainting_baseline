import torch
import os
import cv2
import numpy as np
from skimage.feature import canny
import shutil
import importlib
from .pretty_table import pretty_table
from .ploter import Ploter
import pandas as pd
import csv


def tensor2cv(imgs_t, toRGB = True):
    imgs = []
    for i in range(imgs_t.size(0)):
        im = imgs_t[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        #img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if toRGB:
            im = im[...,::-1]
        imgs.append(im)

    return imgs

def cv2tensor(imgs):
    imgs_t = []
    for im in imgs:
        im_t = torch.from_numpy(im.transpose(2,0,1)).float().div(255.)
        im_t = torch.unsqueeze(im_t,dim=0)
        imgs_t.append(im_t)

    return torch.cat(imgs_t,dim=0)

def imgTensor2edge(imgs_t,sigma=2.,mask=None):
    imgs_np = tensor2cv(imgs_t)
    edge_imgs = []
    for im in imgs_np:
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        edge_im = canny(gray_im, sigma=sigma, mask=mask).astype(np.float)
        edge_im = np.expand_dims(edge_im,axis=-1)
        edge_imgs.append(edge_im)

    edge_imgs = cv2tensor(edge_imgs)
    return edge_imgs.to(imgs_t.device)

def checkDir(dirs):
    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
            except Exception as e:
                pass

def get_file_info(path):
    dir_path, file_full_name = os.path.split(path)
    file_name, file_type = os.path.splitext(file_full_name)

    return {"dir_path": dir_path, "file_full_name": file_full_name,
            "file_name": file_name, "file_type": file_type}


def files_copy(src, dst):
    if os.path.isfile(src):
        shutil.copy(src, dst)

    if os.path.isdir(src):
        shutil.copytree(src, dst)


def find_model_using_name(package_name, model_name,**kwargs):
    modellib = importlib.import_module(package_name)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
    if model !=None:
        return model(**kwargs)
    else:
        print("No model named {model_name} from model zoo!")
        return None

def del_redundant_rows(save_file,keys):
    df = pd.read_csv(save_file)
    # print(df.head())
    indexs = df[df['Model'].isin(keys)].index
    df.drop(index=indexs,axis=0,inplace=True)
    df.to_csv(save_file,index=False)
    # print(df.head())

def save2csv(experiment_name,save_results_dict,
             saveDir='',sortby='FID',update=False):
    if saveDir != '':
        save_file = os.path.join(saveDir,f'{experiment_name}.csv')
    else:
        save_file = f'./results/{experiment_name}.csv'

    if not os.path.exists(save_file):
        update = False

    mode = 'a' if update else 'w'
    if update:
        #delete old item
        del_redundant_rows(save_file,keys=list(save_results_dict.keys()))


    with open(save_file, mode) as f:
        writer = csv.writer(f, dialect='excel')
        write_title_row = True if not update else False
        for k, v in save_results_dict.items():
            print('recording {}...'.format(k))
            if write_title_row:
                writer.writerow(list(v.keys()))
                writer.writerow(list(v.values()))
                write_title_row = False
            else:
                writer.writerow(list(v.values()))
    pretty_table(save_file,title=experiment_name,sortby=sortby)

    ploter = Ploter(save_file)

    return ploter

