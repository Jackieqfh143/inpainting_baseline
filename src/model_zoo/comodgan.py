from src.lib.CoModGAN.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)
import torch
import torch.nn.functional as F
from src.model_zoo.basemodel import BaseModel


class CoModGAN(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize = 256, **kwargs):
        super(CoModGAN, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        if targetSize == 256:
            comodgan_mapping = CoModGANMapping(num_ws=14)
        else:
            comodgan_mapping = CoModGANMapping(num_ws=16)

        comodgan_encoder = CoModGANEncoder(resolution=targetSize)
        comodgan_synthesis = CoModGANSynthesis(resolution=targetSize)
        self.G = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
        self.G.load_state_dict(torch.load(model_path))
        self.G.eval().requires_grad_(False)
        self.G = self.G.to(self.device)

        # no Labels.
        print('CoModGAN loaded.')

    @torch.no_grad()
    def forward(self,imgs, masks, **kwargs):
        input_size = imgs.size(-1)
        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)
        x = torch.cat([masks - 0.5, imgs * masks], dim=1)
        self.inputs = [torch.cat([masks - 0.5, imgs * masks], dim=1)]
        output = self.G(x)
        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        return output


