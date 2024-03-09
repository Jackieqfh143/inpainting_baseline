import torch
from src.lib.CoordFill.models.models import make
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
import yaml

class CoordFill(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=512, config_path = "./src/lib/CoordFill/configs/test/configs.yaml", **kwargs):
        super(CoordFill, self).__init__(**kwargs)
        with open(config_path, 'r') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.G = make(opt['model']).to(self.device)
        self.G.encoder.load_state_dict(torch.load(model_path, map_location=self.device))
        self.G.eval().requires_grad_(False)

        # no Labels.
        print('CoordFill loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        bt,c,h,w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        output = []
        for i in range(bt):
            output.append(self.G.encoder.mask_predict([imgs[i:i+1],masks[i:i+1]]))

        output = torch.cat(output, dim=0)
        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        return output



