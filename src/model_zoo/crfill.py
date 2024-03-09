from src.lib.CrFill.networks.inpaint_g import BaseConvGenerator as generator
import torch
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F


def load_network(net, weights, opt):
    if opt.load_baseg:
        _weights = {}
        for k,v in weights.items():
            if k.startswith("baseg"):
                _k = k.replace("baseg.","")
                _weights[_k] = v
        for k,v in _weights.items():
            weights[k] = v
    new_dict = {}
    for k,v in weights.items():
        if k.startswith("module."):
            k=k.replace("module.","")
        new_dict[k] = v
    net.load_state_dict(new_dict, strict=False)
    return net


class CrFill(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=256, config_path = "./src/lib/CrFill/configs/crfill.yaml", **kwargs):
        super(CrFill, self).__init__(**kwargs)
        self.opt = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize

        self.G = generator(self.opt).to(self.device)
        weights = torch.load(model_path, map_location=self.device)
        self.model = load_network(self.G, weights, self.opt)
        self.G.eval().requires_grad_(False)
        self.one_for_holes = True

        # no Labels.
        print('CrFill loaded.')

    @torch.no_grad()
    def forward(self, imgs, masks,**kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        if self.one_for_holes:
            masks = 1 - masks  # 1 for hole

        masked_imgs = imgs * (1 - masks)
        self.inputs = [masked_imgs, masks]
        _, pred_img = self.G(masked_imgs, masks)
        comp_imgs = (1 - masks) * imgs + masks * pred_img
        output = torch.clamp(comp_imgs, -1, 1)

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        return output



