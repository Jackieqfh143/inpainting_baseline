from src.lib.RFRNet.RFRNetModel import RFRNetModel as RFRNet_model
import torch
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F

class RFRNet(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=256, **kwargs):
        super(RFRNet, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize

        self.G = RFRNet_model()
        state_dict = torch.load(model_path, map_location=self.device)["generator"]
        self.G.load_state_dict(state_dict)
        self.G.eval()
        self.G = self.G.to(self.device)
        # cal params
        print('RFRNet loaded.')

    @torch.no_grad()
    def forward(self, imgs, masks, **kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        imgs = (imgs + 1.0) * 0.5
        masks_3 = torch.cat([masks, masks, masks], dim=1)
        masked_imgs = imgs * masks
        fake, _ = self.G(masked_imgs, masks_3)
        output = fake * (1 - masks_3) + imgs * masks_3

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        output = (output / 0.5) - 1.0
        return output


