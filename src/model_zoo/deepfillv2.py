import cv2
import tensorflow as tf
import neuralgym as ng
import numpy as np
from src.lib.DeepFill_v2.inpaint_model import InpaintCAModel
from src.utils.util import  tensor2cv,cv2tensor
import torch
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
tf.logging.set_verbosity(tf.logging.ERROR)

class DeepFillv2(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=512, config_path = "./src/lib/DeepFill_v2/configs/deepfill_v2.yaml", **kwargs):
        super(DeepFillv2, self).__init__(**kwargs)
        self.opt = self.load_config(config_path)
        self.info = info
        self.device = torch.device(device)
        self.targetSize = targetSize
        self.one_for_holes = True
        self.device_arg = device.replace('cuda', 'GPU')
        FLAGS = ng.Config(config_path)
        self.gc_graph = tf.Graph()
        with self.gc_graph.as_default():
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.gc_sess = tf.Session(config=sess_config, graph=self.gc_graph)

            model = InpaintCAModel()
            im_size = self.targetSize
            self.input_image_ph = tf.placeholder(
                tf.float32, shape=(1, im_size, im_size * 2, 3))
            output = model.build_server_graph(FLAGS, self.input_image_ph)
            output = tf.reverse(output, [-1])
            self.output = tf.saturate_cast(output, tf.uint8)
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(
                    model_path, from_name)
                assign_ops.append(tf.assign(var, var_value))

            self.gc_sess.run(assign_ops)

        # no Labels.
        print('DeepFillv2 loaded.')

    def local_preprocess(self, image, mask):
        h, w, _ = image.shape
        grid = 8
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]
        if image.shape[0] != self.targetSize:
            image = cv2.resize(image, (self.targetSize, self.targetSize),
                               interpolation=cv2.INTER_LINEAR)
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        return input_image

    @torch.no_grad()
    def forward(self, imgs, masks,**kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        if self.one_for_holes:
            masks = 1 - masks  # 1 for hole

        masks = torch.cat([masks, masks, masks], dim = 1)

        imgs = (imgs + 1.0) * 0.5
        imgs_np = tensor2cv(imgs, toRGB=False)
        masks_np = tensor2cv(masks, toRGB=False)

        out_imgs = []
        with tf.device(f'/{self.device_arg}'):
            for img_np, mask_np in zip(imgs_np, masks_np):
                input_img = self.local_preprocess(img_np, mask_np)
                with self.gc_graph.as_default():
                    result = self.gc_sess.run(self.output, feed_dict={self.input_image_ph: input_img})
                out_imgs.append(result[0].copy())

        output = cv2tensor(out_imgs).to(imgs.device)
        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        output = (output / 0.5) + 1.0
        return output
