import torch
from src.lib.TFill.model import networks
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
from src.utils.complexity import get_flops

def load_network(net,state_dict):
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    if isinstance(net, torch.nn.DataParallel):
        net = net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)

    return net

class TFill(BaseModel):

    def __init__(self,model_path, device, info = {}, targetSize=512,
                 config_path = "./src/lib/TFill/configs/tfill.yaml",
                 **kwargs):
        super(TFill, self).__init__(**kwargs)
        self.opt = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.device = torch.device(device)
        self.targetSize = self.targetSize
        self.netE = networks.define_E(self.opt)
        self.netT = networks.define_T(self.opt)
        self.netG = networks.define_G(self.opt)

        opt = self._refine_opt(self.opt)
        self.netG_Ref = networks.define_G(opt)
        self.models = [self.netE, self.netT, self.netG, self.netG_Ref]
        model_names = ['E', 'T', 'G', 'G_Ref']

        for i, item in enumerate(zip(self.models, model_names)):
            net, model_name = item
            weight = torch.load(model_path + f'/latest_net_{model_name}.pth', map_location=self.device)
            self.models[i] = load_network(net, weight)
            self.models[i].eval().requires_grad_(False)
            self.models[i] = self.models[i].to(self.device)

        # cal model params
        print('TFill loaded.')

    def _refine_opt(self, opt):
        """modify the opt for refine generator and discriminator"""
        opt.netG = 'refine'
        opt.netD = 'style'

        return opt

    @torch.no_grad()
    def forward(self,imgs, masks,**kwargs):
        bt,c,h,w = imgs.size()
        img_m = masks * imgs

        fixed_img = F.interpolate(img_m, size=[self.opt.fixed_size, self.opt.fixed_size], mode='bicubic',
                                  align_corners=True).clamp(-1, 1)
        fixed_mask = (F.interpolate(masks, size=[self.opt.fixed_size, self.opt.fixed_size], mode='bicubic',
                                    align_corners=True) > 0.9).type_as(fixed_img)

        self.inputs1 = [fixed_img, fixed_mask]
        out, mask = self.netE(fixed_img, mask=fixed_mask, return_mask=True)

        self.inputs2 = [out, mask]
        out = self.netT(out, mask, bool_mask=False)

        self.inputs3 = [out, masks]
        img_g = self.netG(out, mask=masks)

        img_g_org = F.interpolate(img_g, size=imgs.size()[2:], mode='bicubic', align_corners=True).clamp(-1,1)
        img_out = masks * imgs + (1 - masks) * img_g_org

        self.inputs4 = [img_out, masks]
        img_ref = self.netG_Ref(img_out, mask=masks)
        output = masks * imgs + (1 - masks) * img_ref

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        return output

    def get_complex(self):
        flops,param = 0.0, 0.0
        tmp = get_flops(self.netE, self.inputs1)
        flops += tmp[0]
        param += tmp[1]

        tmp = get_flops(self.netT, self.inputs2)
        flops += tmp[0]
        param += tmp[1]

        tmp = get_flops(self.netG, self.inputs3)
        flops += tmp[0]
        param += tmp[1]

        tmp = get_flops(self.netG_Ref, self.inputs4)
        flops += tmp[0]
        param += tmp[1]

        print("FLOPs: ", flops)
        print("Param: ", param)

        return flops, param



