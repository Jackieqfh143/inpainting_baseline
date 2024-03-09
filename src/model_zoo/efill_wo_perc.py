import torch
from src.lib.EFill.modules.models import DistInpaintModel_SPADE_IN_LFFC_Base_concat_WithAtt as EFill_Model
from collections import OrderedDict
import torch.nn.functional as F
from src.model_zoo.basemodel import BaseModel

class EFill_WO_PERC(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=256,**kwargs):
        super(EFill_WO_PERC, self).__init__(**kwargs)
        self.info = info
        self.targetSize = targetSize
        self.device = torch.device(device)
        self.G = EFill_Model(is_training=False)
        net_state_dict = self.G.state_dict()
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
        self.G.load_state_dict(OrderedDict(new_state_dict), strict=False)
        self.G.eval().requires_grad_(False).to(device)
        self.edge_type = "sobel"

        # no Labels.
        print('EFill_WO_PERC loaded.')

    @torch.no_grad()
    def forward(self,imgs, masks, edges, **kwargs):
        input_size = imgs.size(-1)
        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)
            edges = F.interpolate(edges, self.targetSize, mode="bilinear")
        masked_im = imgs * masks
        masked_edge = edges * masks
        input_x = torch.cat((masked_im, masks, masked_edge), dim=1)
        output,*_ = self.G(input_x)
        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        output = (output / 0.5) - 1.0
        return output

