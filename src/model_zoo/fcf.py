import src.lib.FcF.dnnlib as dnnlib
import torch
import src.lib.FcF.legacy as legacy
import torch.nn.functional as F
from src.model_zoo.basemodel import BaseModel

class FcF(BaseModel):

    def __init__(self,model_path, device, info = {}, targetSize=256, **kwargs):
        super(FcF, self).__init__(**kwargs)
        self.device = torch.device(device)
        self.targetSize = targetSize
        with dnnlib.util.open_url(model_path) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False)  # type: ignore
        self.info = info
        # Labels.
        self.label = torch.zeros([1, self.G.c_dim], device=self.device)
        class_idx = None
        if self.G.c_dim != 0:
            if class_idx is None:
                print('Must specify class label with --class when using a conditional network')
            self.label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')

        print('FcF loaded.')

    @torch.no_grad()
    def forward(self,real_imgs,masks, **kwargs):
        input_size = real_imgs.size(-1)
        if input_size != self.targetSize:
            real_imgs = F.interpolate(real_imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)
        masks = 1 - masks    #1 for holes
        img_masked = real_imgs.clone()
        img_masked = img_masked * (1 - masks)
        img_masked = img_masked.to(torch.float32)

        self.inputs = [torch.cat([0.5 - masks, img_masked], dim=1), self.label]
        pred_im = self.G(img=torch.cat([0.5 - masks, img_masked], dim=1), c=self.label, truncation_psi=0.1,
                     noise_mode='const')
        comp_imgs = masks * pred_im + (1 - masks) * real_imgs

        if comp_imgs.size(-1) != input_size:
            comp_imgs = F.interpolate(comp_imgs, input_size, mode="bilinear")

        return comp_imgs


