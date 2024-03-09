import os

from src.lib.MobileFill.src.models.mobileFill import MobileFill as InpaintModel
import torch.nn.functional as F
from src.model_zoo.basemodel import BaseModel
import torch
from collections import OrderedDict
import yaml
import glob

class MobileFill_ONLY_RAND(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=256, config_path = "./src/lib/MobileFill/configs/ablations/only_randmask.yaml", **kwargs):
        super(MobileFill_ONLY_RAND, self).__init__(**kwargs)
        with open(config_path, "r") as f:
            opts = yaml.load(f, yaml.FullLoader)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize

        self.G = InpaintModel(**opts).to(self.device)

        if os.path.isdir(model_path):
            self.state_dict_list = []
            model_path = sorted(glob.glob(model_path + "/*.pth"))
            for path in model_path:
                net_state_dict = self.G.state_dict()
                state_dict = torch.load(path, map_location=device)["ema_G"]
                new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
                self.state_dict_list.append(OrderedDict(new_state_dict))

            self.G.load_state_dict(self.state_dict_list[0])
        else:
            net_state_dict = self.G.state_dict()
            state_dict = torch.load(model_path, map_location='cpu')["ema_G"]
            new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
            self.G.load_state_dict(OrderedDict(new_state_dict), strict=False)

        self.G.eval().requires_grad_(False).to(device)
        # no Labels.
        print('MobileFill_ONLY_RAND loaded.')

    @torch.no_grad()
    def forward(self, imgs, masks,**kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        masked_im = imgs * masks  # 0 for holes
        input_x = torch.cat((masked_im, masks), dim=1)
        self.inputs = [input_x]
        pred_img = self.G(input_x)[0]["img"]
        output = (1 - masks) * pred_img + imgs * masks

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        return output










