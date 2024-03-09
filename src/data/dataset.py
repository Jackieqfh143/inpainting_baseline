import traceback

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torchvision.transforms.functional as F
import glob
import random
import numpy as np
from PIL import Image
import cv2
from skimage.feature import canny


def resize(img, target_size, aspect_ratio_kept = True,
           fixed_size = True, center_crop = True):
    if aspect_ratio_kept:
        imgh, imgw = img.shape[0:2]
        side = np.minimum(imgh, imgw)
        if fixed_size:
            if center_crop:
                # center crop
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img[j:j + side, i:i + side, ...]
            else:
                #random crop
                j = (imgh - side)
                i = (imgw - side)
                h_start = 0
                w_start = 0
                if j != 0:
                    h_start = random.randrange(0, j)
                if i != 0:
                    w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
        else:
            if side <= target_size:
                j = (imgh - side)
                i = (imgw - side)
                h_start = 0
                w_start = 0
                if j != 0:
                    h_start = random.randrange(0, j)
                if i != 0:
                    w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                side = random.randrange(target_size, side)
                j = (imgh - side)
                i = (imgw - side)
                h_start = random.randrange(0, j)
                w_start = random.randrange(0, i)
                img = img[h_start:h_start + side, w_start:w_start + side, ...]
    img = np.array(Image.fromarray(img).resize(size=(target_size, target_size)))
    return img

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ValDataSet(data.Dataset):
    def __init__(self,img_dir,mask_dir,total_num):
        super(ValDataSet, self).__init__()
        self.imgs = sorted(glob.glob(img_dir + "/*.jpg") + glob.glob(img_dir + "/*.png"))
        self.masks = sorted(glob.glob(mask_dir + "/*.jpg") + glob.glob(mask_dir + "/*.png"))


        max_num = min(len(self.imgs),total_num)
        self.imgs = self.imgs[:max_num]
        self.masks = self.masks[:max_num]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            traceback.print_exc()
            print('loading error: ' + self.imgs[index])
            item = self.load_item(0)

        return item

    def load_item(self,idx):
        img_256,img_512 = self.load_multi_imgs(self.imgs[idx])
        mask_256, mask_512 = self.load_multi_imgs(self.masks[idx], resize_type=Image.Resampling.NEAREST)
        item_256 = self.preprocess(img_256, mask_256)
        item_512 = self.preprocess(img_512, mask_512)
        return item_256, item_512

    def load_multi_imgs(self, img_path, resize_type = Image.Resampling.BILINEAR):
        img = Image.open(img_path)
        if img.size[-1] == 512:
            img_256 = np.array(img.resize((256, 256),resize_type))
            img_512 = np.array(img)

        elif img.size[-1] == 256:
            img_512 = np.array(img.resize((512, 512), resize_type))
            img_256 = np.array(img)

        else:
            img_512 = np.array(img.resize((512, 512), resize_type))
            img_256 = np.array(img.resize((256, 256), resize_type))

        return img_256, img_512


    #propocess one image each time
    def preprocess(self,img,mask):
        edge, gray = self.load_edge(img)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.concatenate((mask, mask, mask), axis=-1)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate((img, img, img), axis=-1)

        mask = mask.astype(np.uint8)

        mask = (mask > 0).astype(np.uint8) * 255
        img_t_raw = F.to_tensor(img).float()
        gray_t = F.to_tensor(gray).float()
        edge_t = F.to_tensor(edge).float()
        mask_t = F.to_tensor(mask).float()

        mask_t = mask_t[2:3,:,:]
        mask_t = 1 - mask_t    #set holes = 0
        img_t = img_t_raw / 0.5 - 1.0
        gray_t = gray_t / 0.5 - 1.0

        return img_t, gray_t, edge_t, mask_t

    def load_edge(self, img, edge_type="sobel"):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if edge_type == "sobel":
            x = cv2.Sobel(gray, -1, 1, 0, ksize=3, scale=1)
            y = cv2.Sobel(gray, -1, 0, 1, ksize=3, scale=1)
            absx = cv2.convertScaleAbs(x)
            absy = cv2.convertScaleAbs(y)
            edge = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        else:
            edge = canny(gray, sigma=2., mask=None).astype(np.float)
            edge = cv2.convertScaleAbs(edge) * 255

        return edge, gray

class Configurations():
    def __init__(self):
        self.checkpoints_info = {}
        self.paper_info = {}