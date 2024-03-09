import torch
from src.lib.MIGAN.migan_inference import Generator as MIGAN_Model
import torch.nn.functional as F
from src.model_zoo.basemodel import BaseModel

class MIGAN(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=256,**kwargs):
        super(MIGAN, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.G = MIGAN_Model(resolution=targetSize)
        self.G.load_state_dict(torch.load(model_path))
        self.G.eval().requires_grad_(False)
        self.G = self.G.to(self.device)

        # no Labels.
        print('MIGAN loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        input_size = imgs.size(-1)
        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)
        x = torch.cat([masks - 0.5, imgs * masks], dim=1)
        output = self.G(x)
        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        return output
