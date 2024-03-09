from src.lib.MADF.net import MADFNet
from src.lib.MADF.util.io import load_ckpt as load_MADF
from src.model_zoo.basemodel import BaseModel
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import torch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
class MADF(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=512,**kwargs):
        super(MADF, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.G = MADFNet(layer_size=7, n_refinement_D=2)
        load_MADF(model_path, [('model', self.G)])
        self.G.eval()
        self.G = self.G.to(self.device)

        self.img_transform = transforms.Compose(
            [transforms.Resize(size=self.targetSize),
             transforms.Normalize(mean=MEAN, std=STD)])
        self.mask_transform = transforms.Compose(
            [transforms.Resize(size=self.targetSize, interpolation=Image.NEAREST)])

        self.std = torch.tensor(STD, device = self.device)
        self.mean = torch.tensor(MEAN, device = self.device)

        # no Labels.
        print('MADF loaded.')

    def local_preprocess(self, imgs, masks):
        img_trans = []
        mask_trans = []
        for img, mask in zip(imgs, masks):
            img_tran = self.img_transform(img)
            mask_tran = self.mask_transform(mask)
            img_tran = img_tran.unsqueeze(0)
            mask_tran = mask_tran.unsqueeze(0)
            img_trans.append(img_tran)
            mask_trans.append(mask_tran)

        img_trans = torch.cat(img_trans, dim=0)
        mask_trans = torch.cat(mask_trans, dim=0)
        return img_trans, mask_trans

    def unnormalize(self, x):
        x = x.transpose(1, 3)
        x = x * self.std + self.mean
        x = x.transpose(1, 3)
        return x

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        input_size = imgs.size(-1)
        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        imgs = (imgs + 1.0) * 0.5
        masks = torch.cat([masks, masks, masks], dim=1)
        real_imgs_norm, masks_norm = self.local_preprocess(imgs, masks)
        input_image = real_imgs_norm * masks_norm

        outputs = self.G(input_image, masks_norm)
        output = outputs[-1]
        comp_imgs_norm = input_image * masks_norm + (1 - masks_norm) * output
        comp_imgs = []
        for i in range(comp_imgs_norm.shape[0]):
            comp_img_norm = comp_imgs_norm[i:i + 1]
            comp_img = self.unnormalize(comp_img_norm)
            comp_imgs.append(comp_img)

        comp_imgs = torch.cat(comp_imgs, dim=0)
        comp_imgs = (comp_imgs / 0.5) - 1.0

        if comp_imgs.size(-1) != input_size:
            comp_imgs = F.interpolate(comp_imgs, input_size, mode="bilinear")

        return comp_imgs