from src.lib.HiFill.model import get_HiFill_output
from src.model_zoo.basemodel import BaseModel
import torch
import tensorflow as tf
import torch.nn.functional as F
from src.utils.util import tensor2cv,cv2tensor

class HiFill(BaseModel):

    def __init__(self,model_path,device, info = {}, targetSize=512,**kwargs):
        super(HiFill, self).__init__(**kwargs)
        self.info = info
        self.device_arg = device.replace('cuda', 'GPU')
        self.targetSize = targetSize
        self.hf_graph = tf.Graph()
        with self.hf_graph.as_default():
            with open(model_path, "rb") as f:
                output_graph_def = tf.GraphDef()
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.hf_sess = tf.Session(config=sess_config, graph=self.hf_graph)
            init = tf.global_variables_initializer()
            self.hf_sess.run(init)
            self.hf_image_ph = self.hf_sess.graph.get_tensor_by_name('img:0')
            self.hf_mask_ph = self.hf_sess.graph.get_tensor_by_name('mask:0')
            self.hf_inpainted_512_node = self.hf_sess.graph.get_tensor_by_name('inpainted:0')
            self.hf_attention_node = self.hf_sess.graph.get_tensor_by_name('attention:0')
            self.hf_mask_512_node = self.hf_sess.graph.get_tensor_by_name('mask_processed:0')

        # no Labels.
        print('HiFill loaded.')

    @torch.no_grad()
    def forward(self,imgs,masks,**kwargs):
        input_size = imgs.size(-1)

        if input_size != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        args = [self.hf_graph, self.hf_sess, self.hf_image_ph, self.hf_mask_ph, self.hf_inpainted_512_node,
                self.hf_attention_node, self.hf_mask_512_node]

        imgs =  (imgs + 1.0) * 0.5
        imgs_np = tensor2cv(imgs, toRGB=False)
        masks_t = torch.cat((masks, masks, masks), dim=1)
        masks_np = tensor2cv(masks_t, toRGB=False)
        comp_imgs = []
        with tf.device(f'/{self.device_arg}'):
            for im, mask in zip(imgs_np, masks_np):
                fake_im = get_HiFill_output(im, mask, *args)
                comp_imgs.append(fake_im)

        output = cv2tensor(comp_imgs).to(imgs.device)
        if output.size(-1) != input_size:
            output = F.interpolate(output, input_size, mode="bilinear")

        output = (output / 0.5) - 1.0
        return output
