from src.lib.AoTGAN.aotgan import InpaintGenerator as AOT_model
import torch.nn.functional as F
from src.model_zoo.basemodel import BaseModel
import torch

class AoTGAN(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=512,**kwargs):
        super(AoTGAN, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.G = AOT_model()
        self.G.load_state_dict(torch.load(model_path))
        self.G.eval().requires_grad_(False)
        self.G = self.G.to(self.device)
        self.one_for_hole = True

        # no Labels.
        print('AoTGAN loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        input_size = imgs.size(-1)
        if self.one_for_hole:
            masks = 1 - masks

        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        masked_imgs = imgs * (1 - masks) + masks
        self.inputs = [masked_imgs, masks]
        output = self.G(masked_imgs, masks)
        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        return output










