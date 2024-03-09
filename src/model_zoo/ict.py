import torch
from src.lib.ICT.Guided_Upsample.src.Guided_Upsampler import Guided_Upsampler
from src.lib.ICT.Transformer.models.model import GPTConfig,GPT
from src.lib.ICT.Transformer.utils.util import sample_mask
from src.lib.ICT.Guided_Upsample.src.models import InpaintingModel
from src.lib.ICT.Guided_Upsample.src.networks import InpaintGenerator_5
import numpy as np
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F


class ICT(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=512, config_path = "./src/lib/ICT/configs/ict.yaml",
                 cluster_path = "./src/lib/ICT/Guided_Upsample/kmeans_centers.npy", **kwargs):
        super(ICT, self).__init__(**kwargs)
        self.opts = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        model_config = GPTConfig(512, self.targetSize * self.targetSize,
                                 embd_pdrop=0.0, resid_pdrop=0.0,
                                 attn_pdrop=0.0, n_layer=self.opts.n_layer, n_head=self.opts.n_head,
                                 n_embd=self.opts.n_embd, BERT=self.opts.BERT, use_gelu2=self.opts.GELU_2)

        # Load transfomer model
        self.IGPT_model = GPT(model_config)
        checkpoint = torch.load(model_path, map_location='cpu')

        if model_path.endswith('.pt'):
            self.IGPT_model.load_state_dict(checkpoint)
        else:
            self.IGPT_model.load_state_dict(checkpoint['model'])

        # Load inpainting network
        self.G = InpaintGenerator_5()
        data = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(data['generator'])
        self.G.eval().requires_grad_(False)

        # cal params
        # Load transformer clusters
        C = np.load(cluster_path)  ## [0,1]
        C = np.rint(127.5 * (C + 1.0))
        self.C = torch.from_numpy(C)

        self.one_for_holes = True

        # no Labels.
        print('ICT loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        bt,c,h,w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        if self.one_for_holes:
            masks = 1 - masks

        comp_imgs = []
        n_samples = 1
        real_imgs = (imgs + 1.0) * 0.5
        for i in range(real_imgs.shape[0]):
            # get appearance edge priors from pretrained transformer
            x = real_imgs[i].view(-1, 3)
            x = x.float()
            a = ((x[:, None, :] - self.C[None, :, :]) ** 2).sum(-1).argmin(1)  # cluster assignments

            y = masks[i].view(-1)
            y = y > 0.5
            y = y.float()

            a_list = [a] * n_samples
            a_tensor = torch.stack(a_list, dim=0)  ## Input images
            b_list = [y] * n_samples
            b_tensor = torch.stack(b_list, dim=0)  ## Input masks
            a_tensor *= (1 - b_tensor).long()

            pixels = sample_mask(self.IGPT_model, context=a_tensor, length=self.opts.image_size * self.opts.image_size,
                                 num_sample=n_samples, top_k=self.opts.top_k, mask=b_tensor, no_bar=True)

            edge_priors = []
            for i in range(n_samples):
                current_prior = self.C[pixels[i]].view(self.opts.image_size, self.opts.image_size, 3).numpy().astype(
                    np.uint8)
                edge_priors.append(current_prior)

            # generate multiple outputs
            for edge in edge_priors:
                # inpainting process
                image_masked = (real_imgs[i:i + 1] * (1 - masks[i:i + 1]).float()) + masks[i:i + 1]
                input = torch.cat((image_masked, edge), dim=1)
                output = self.G(input)
                comp_imgs.append(output)

        output = torch.cat(comp_imgs, dim=0)
        output = (output / 0.5) - 1.0
        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        return output



