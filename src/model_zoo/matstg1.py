from src.lib.MAT.networks.mat import Generator
import src.lib.MAT.dnnlib as dnnlib
import src.lib.MAT.legacy as legacy
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
import torch

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


class MATSTG1(BaseModel):

    def __init__(self,model_path, device, info = {}, targetSize=256,**kwargs):
        super(MATSTG1, self).__init__(**kwargs)
        self.device = torch.device(device)
        with dnnlib.util.open_url(model_path) as f:
            G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
        self.G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=targetSize, img_channels=3).eval().requires_grad_(False)
        copy_params_and_buffers(G_saved, self.G, require_all=True)
        self.G = self.G.to(self.device)
        self.info = info
        self.targetSize = targetSize

        # no Labels.
        self.label = torch.zeros([1, self.G.c_dim], device=self.device)
        print('MATSTG1 loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        input_size = imgs.size(-1)
        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)
        noise_mode = 'random'
        z = torch.randn(imgs.size(0), self.G.z_dim).to(self.device)
        _, _, stg1 = self.G(imgs, masks, z, self.label, truncation_psi=1, noise_mode=noise_mode)
        if stg1.size(-1) != input_size:
            stg1 = F.interpolate(stg1, input_size, mode="bilinear")

        return stg1


