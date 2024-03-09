from src.lib.PIC.util import task as PL_task
from src.lib.PIC.util.util import load_network as load_pl_model
from src.lib.PIC.model import network
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
import torch
from src.utils.complexity import get_flops


class PIC(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=512,**kwargs):
        super(PIC, self).__init__(**kwargs)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.net_E = network.define_e(ngf=32, z_nc=128, img_f=128, layers=5, norm='none', activation='LeakyReLU',
                                      init_type='orthogonal', gpu_ids=[0])
        self.net_G = network.define_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=4,
                                      norm='instance', activation='LeakyReLU', init_type='orthogonal', gpu_ids=[0])

        self.net_E = load_pl_model(self.net_E, model_path + '/latest_net_E.pth')
        self.net_G = load_pl_model(self.net_G, model_path + '/latest_net_G.pth')
        self.net_E.eval().requires_grad_(False)
        self.net_E = self.net_E.to(self.device)
        self.net_G = self.net_G.to(self.device)

        # no Labels.
        print('AoTGAN loaded.')

    @torch.no_grad()
    def forward(self, imgs, masks, **kwargs):
        input_size = imgs.size(-1)

        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        img_masked = imgs * masks

        self.inputs1 = [img_masked]
        # encoder process
        distribution, f = self.net_E(img_masked)
        q_distribution = torch.distributions.Normal(distribution[-1][0], distribution[-1][1])
        scale_mask = PL_task.scale_img(masks, size=[f[2].size(2), f[2].size(3)])

        z = q_distribution.sample()
        self.inputs2 = [z, f[-1], f[2], scale_mask.chunk(3, dim=1)[0]]
        img_g, attn = self.net_G(z, f_m=f[-1], f_e=f[2], mask=scale_mask.chunk(3, dim=1)[0])
        output = (1 - masks) * img_g[-1] + masks * imgs

        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        return output

    def get_complex(self):
        flops, param = 0.0, 0.0
        tmp = get_flops(self.net_E, self.inputs1)
        flops += tmp[0]
        param += tmp[1]

        tmp = get_flops(self.net_G, self.inputs2)
        flops += tmp[0]
        param += tmp[1]

        print("FLOPs: ", flops)
        print("Param: ", param)

        return flops, param





