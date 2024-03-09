import numpy as np

from src.lib.SPL.models_inpaint import InpaintingModel as SPL_model
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
from torchvision.transforms import functional as transF
from PIL import Image
import torch
import cv2
from src.utils.util import tensor2cv
from src.utils.complexity import get_flops

class SPL(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=256,**kwargs):
        super(SPL, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize

        self.G = SPL_model(g_lr=0, d_lr=0, l1_weight=0, gan_weight=0, iter=0,
                          threshold=0.8)
        pretrained_model = torch.load(model_path, map_location=self.device)
        state_dict = self.G.state_dict()
        new_dict_no_module = {}
        for k, v in pretrained_model.items():
            k = k.replace('module.', '')
            new_dict_no_module[k] = v

        new_dict = {k: v for k, v in new_dict_no_module.items() if k in state_dict.keys()}
        state_dict.update(new_dict)
        self.G.load_state_dict(state_dict)

        self.G.eval().requires_grad_(False)
        self.G = self.G.to(self.device)
        self.one_for_hole = True

        # no Labels.
        print('SPL loaded.')

    def to_tensor(self, imgs):
        img_ts = []
        for img in imgs:
            img = Image.fromarray(img)
            img_t = transF.to_tensor(img).float()
            img_ts.append(img_t)

        return torch.stack(img_ts, dim=0).to(self.device)

    def local_preprocess(self, imgs, masks):
        imgs_512 = []
        imgs_256 = []
        masks_out = []
        for img, mask in zip(imgs, masks):
            img_512 = cv2.resize(img, (512, 512))
            img_256 = cv2.resize(img, (self.targetSize, self.targetSize))

            mask = cv2.resize(mask, (self.targetSize, self.targetSize),
                              interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis = -1)
            mask = np.concatenate((mask,mask,mask), axis=-1)
            imgs_512.append(img_512)
            imgs_256.append(img_256)
            masks_out.append(mask)

        return self.to_tensor(imgs_512), self.to_tensor(imgs_256), self.to_tensor(masks_out)

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        input_size = imgs.size(-1)
        imgs = (imgs + 1.0) * 0.5

        if self.one_for_hole:
            masks = 1 - masks

        imgs_np = tensor2cv(imgs, toRGB=False)
        masks_np = tensor2cv(masks, toRGB=False)

        gt_512_batch, gt_batch, mask_batch = self.local_preprocess(imgs_np, masks_np)
        mask_batch = torch.mean(mask_batch, 1, keepdim=True)
        mask_512 = F.interpolate(mask_batch, 512)
        gt_512_masked = gt_512_batch * (1.0 - mask_512) + mask_512

        self.inputs = [gt_batch, mask_batch, gt_512_masked, mask_512]
        prediction, _ = self.G.generator(gt_batch, mask_batch, gt_512_masked, mask_512)
        output = prediction * mask_batch + gt_batch * (1 - mask_batch)

        output = (output / 0.5) - 1.0
        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        return output

    def get_complex(self):
        return get_flops(self.G.generator, self.inputs)