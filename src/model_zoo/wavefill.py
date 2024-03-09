from src.lib.WaveFill.models.networks.generator import WaveletInpaintLv2GCFRANGenerator
import torch
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F


class WaveFill(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=512, config_path = "./src/lib/WaveFill/configs/wavefill.yaml", **kwargs):
        super(WaveFill, self).__init__(**kwargs)
        self.opt = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize

        self.G = WaveletInpaintLv2GCFRANGenerator(self.opt).to(self.device)
        weights = torch.load(model_path, map_location=self.device)
        self.G.load_state_dict(weights, strict=False)
        self.G.eval().requires_grad_(False)
        self.one_for_holes = True

        # no Labels.
        print('WaveFill loaded.')

    @torch.no_grad()
    def forward(self, imgs, masks,**kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        if self.one_for_holes:
            masks = 1 - masks  # 1 for hole

        masked_im = (imgs * (1 - masks)) + masks
        masked_im = torch.cat([masked_im, masks], dim=1)
        generated = self.G(masked_im, z=None)
        output = masked_im[:, :3] * (1 - masks) + generated[-1] * masks

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        return output