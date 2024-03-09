from src.lib.MISF.src.networks import InpaintGenerator
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
import torch

class MISF(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=512, config_path = "./src/lib/MISF/configs/misf.yaml", **kwargs):
        super(MISF, self).__init__(**kwargs)
        self.opt = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize

        self.G = InpaintGenerator(config=self.opt).to(self.device)
        data = torch.load(model_path, map_location=self.device)
        self.G.load_state_dict(data['generator'])
        self.G.eval().requires_grad_(False)
        self.one_for_holes = True

        # no Labels.
        print('MISF loaded.')

    @torch.no_grad()
    def forward(self, imgs, masks,**kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        if self.one_for_holes:
            masks = 1 - masks  # 1 for hole

        imgs = (imgs + 1.0) * 0.5
        images_masked = imgs * (1 - masks)
        inputs = torch.cat((images_masked, masks), dim=1)
        self.inputs = [inputs]
        outputs = self.G(inputs)  # in: [rgb(3) + edge(1)]
        output = (outputs * masks) + (imgs * (1 - masks))
        output = (output / 0.5) - 1.0

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        return output




