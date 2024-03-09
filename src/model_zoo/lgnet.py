from src.lib.LGNet.models import networks
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
import torch
from src.utils.complexity import get_flops


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)


def load_network(net,state_dict):

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        __patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)

    return net

class LGNet(BaseModel):

    def __init__(self,model_path, device, info = {}, targetSize=256,
                 config_path = "./src/lib/LGNet/configs/lgnet.yaml",
                 **kwargs):
        super(LGNet, self).__init__(**kwargs)
        self.opt = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.device = torch.device(device)
        self.targetSize = self.targetSize

        # define networks (both generator and discriminator)
        self.netG1 = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG1, self.opt.norm,
                                       not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, gpu_ids=[])
        self.netG2 = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG2, self.opt.norm,
                                       not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, gpu_ids=[])
        self.netG3 = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG3, self.opt.norm,
                                       not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, gpu_ids=[])

        weight_G1 = torch.load(model_path + '/latest_net_G1.pth', map_location=self.device)
        weight_G2 = torch.load(model_path + '/latest_net_G2.pth', map_location=self.device)
        weight_G3 = torch.load(model_path + '/latest_net_G3.pth', map_location=self.device)

        self.models = [self.netG1, self.netG2, self.netG3]
        weight_list = [weight_G1, weight_G2, weight_G3]

        for i, item in enumerate(zip(self.models, weight_list)):
            self.models[i] = load_network(*item)
            self.models[i].eval().requires_grad_(False)
            self.models[i] = self.models[i].to(self.device)


        self.one_for_holes = True

        print('LGNet loaded.')

    def _refine_opt(self, opt):
        """modify the opt for refine generator and discriminator"""
        opt.netG = 'refine'
        opt.netD = 'style'

        return opt

    @torch.no_grad()
    def forward(self,imgs, masks, **kwargs):
        if self.one_for_holes:
            masks = 1 - masks  # 1 for hole

        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        data = {'A': imgs, 'B': masks, 'A_paths': ''}
        AtoB = self.opt.direction == 'AtoB'
        images = data['A' if AtoB else 'B']
        masks = data['B' if AtoB else 'A']

        masked_images1 = images * (1 - masks) + masks
        self.inputs1 = [torch.cat((masked_images1, masks), 1)]
        output_images1 = self.netG1(torch.cat((masked_images1, masks), 1))
        merged_images1 = images * (1 - masks) + output_images1 * masks
        self.inputs2 = [torch.cat((merged_images1, masks), 1)]
        output_images2 = self.netG2(torch.cat((merged_images1, masks), 1))
        merged_images2 = images * (1 - masks) + output_images2 * masks
        self.inputs3 = [torch.cat((merged_images2, masks), 1)]
        output_images3 = self.netG3(torch.cat((merged_images2, masks), 1))
        output = images * (1 - masks) + output_images3 * masks

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")


        return output

    def get_complex(self):
        flops, param = 0.0,0.0
        for i in range(1,4):
            tmp = get_flops(getattr(self, f"netG{i}"), getattr(self, f"inputs{i}"))
            flops += tmp[0]
            param += tmp[1]

        print("FLOPs: ", flops)
        print("Param: ", param)
        return flops, param
