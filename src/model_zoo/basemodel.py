import torch
import numpy as np
import torchvision.transforms.functional as F
from skimage.feature import canny
from omegaconf import OmegaConf
from PIL import Image
from src.utils.complexity import get_flops
import cv2
import yaml

class BaseModel():

    def __init__(self, device = "cuda", edge_type = "sobel", target_size = 256,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.edge_type = edge_type
        self.device = device
        self.targetSize = target_size
        self.time_cost = 0.0


    def load_config(self, conf_path):
        with open(conf_path, 'r') as f:
            opt = OmegaConf.create(yaml.safe_load(f))

        return opt


    def preprocess(self,img,mask):
        if isinstance(img, str):
            img = np.array(Image.open(img))

        if isinstance(mask, str):
            mask = np.array(Image.open(mask))

        edge, gray = self.load_edge(img, self.edge_type)
        img = np.array(Image.fromarray(img).resize((self.targetSize, self.targetSize), Image.Resampling.BILINEAR))
        edge = np.array(Image.fromarray(edge).resize((self.targetSize, self.targetSize), Image.Resampling.BILINEAR))
        gray = np.array(Image.fromarray(gray).resize((self.targetSize, self.targetSize), Image.Resampling.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((self.targetSize, self.targetSize), Image.Resampling.NEAREST))

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.concatenate((mask, mask, mask), axis=-1)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.concatenate((img, img, img), axis=-1)

        mask = mask.astype(np.uint8)

        mask = (mask > 0).astype(np.uint8) * 255
        img_t_raw = F.to_tensor(img).float().to(self.device)
        gray_t = F.to_tensor(gray).float().to(self.device)
        edge_t = F.to_tensor(edge).float().to(self.device)
        mask_t = F.to_tensor(mask).float().to(self.device)

        mask_t = mask_t[2:3, :, :]
        mask_t = 1 - mask_t  # set holes = 0
        img_t = img_t_raw / 0.5 - 1.0
        gray_t = gray_t / 0.5 - 1.0

        return img_t.unsqueeze(0), gray_t.unsqueeze(0), edge_t.unsqueeze(0), mask_t.unsqueeze(0)

    def postprocess(self,img):
        img = (img.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()

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

    def get_complex(self):
        return get_flops(self.G, self.inputs)