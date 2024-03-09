from src.lib.EC.src.models import EdgeModel
from src.lib.EC.src.models import InpaintingModel as EC_model
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
import torch

class EC(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=512,
                 config_path = "./src/lib/EC/configs/ec.yaml" , **kwargs):
        super(EC, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.opt = self.load_config(config_path)
        self.edge_model = EdgeModel(self.opt)
        self.inpaint_model = EC_model(self.opt)
        edge_stateDic = torch.load(model_path + '/EdgeModel_gen.pth', map_location='cpu')
        inpaint_stateDic = torch.load(model_path + '/InpaintingModel_gen.pth', map_location='cpu')
        self.edge_model.generator.load_state_dict(edge_stateDic['generator'])
        self.inpaint_model.generator.load_state_dict(inpaint_stateDic['generator'])
        self.edge_model.generator.eval().requires_grad_(False).to(self.device)
        self.inpaint_model.generator.eval().requires_grad_(False).to(self.device)
        self.edge_type = "canny"
        self.one_for_hole = True

        # no Labels.
        print('EC loaded.')

    @torch.no_grad()
    def forward(self,imgs, masks, grays, edges, **kwargs):
        input_size = imgs.size(-1)
        if self.one_for_hole:
            masks = 1 - masks

        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            grays = F.interpolate(grays, self.targetSize, mode="bilinear")
            edges = F.interpolate(edges, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        edges = self.edge_model(grays, edges, masks).detach()
        fake = self.inpaint_model(imgs, edges, masks)
        output = (fake * masks) + (imgs * (1 - masks))


        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        return output