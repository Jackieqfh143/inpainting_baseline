from src.lib.LaMa.saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
import torch

class LaMa(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=512, config_path = "./src/lib/LaMa/configs/lama.yaml", **kwargs):
        super(LaMa, self).__init__(**kwargs)
        self.opt = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.opt.training_model.predict_only = True
        self.opt.visualizer.kind = 'noop'
        kwargs = dict(self.opt.training_model)
        kwargs.pop('kind')
        kwargs['use_ddp'] = self.opt.trainer.kwargs.get('accelerator', None) == 'ddp'
        self.G = DefaultInpaintingTrainingModule(self.opt, **kwargs).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.G.load_state_dict(state['state_dict'], strict=False)
        self.G.on_load_checkpoint(state)
        self.G.freeze()
        self.one_for_holes = True

        # no Labels.
        print('LaMa loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        bt, c, h, w = imgs.size()
        if self.one_for_holes:
            masks = 1 - masks

        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        imgs = (imgs + 1.0) * 0.5
        lama_input = {'image': imgs, 'mask': masks}
        lama_output = self.G(lama_input)
        output = lama_output['inpainted']

        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        output = (output / 0.5) - 1.0
        return output