from src.lib.SKI.src.inpaint_model import InpaintModel
import tensorflow as tf
import numpy as np
from src.utils.util import tensor2cv,cv2tensor
from skimage import feature
from skimage.color import rgb2gray
import os
import torch
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
tf.logging.set_verbosity(tf.logging.ERROR)
def inverse_transform(images):
    return ((images + 1.)*127.5).astype(np.uint8)

def image_edge_processing(img,sigma=1.5):
    # edge
    # gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = rgb2gray(img)  # with the channel dimension removed
    edge = feature.canny(img_gray, sigma=sigma).astype(np.float32)
    # edge = np.expand_dims(edge,axis=-1)
    img = img.astype(np.float32) / 127.5 - 1  # scale to [-1, 1]
    return img, edge


class SKI(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=256, config_path = "./src/lib/SKI/configs/ski.yaml", **kwargs):
        super(SKI, self).__init__(**kwargs)
        self.device_arg = device.replace('cuda', 'GPU')
        self.opt = self.load_config(config_path)
        self.info = info
        self.targetSize = targetSize
        self.one_for_holes = True
        opt = self.load_config(config_path)
        self.sk_graph = tf.Graph()
        with self.sk_graph.as_default():
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sk_sess = tf.Session(config=sess_config, graph=self.sk_graph)
            self.G = InpaintModel(opt)
            self.img_ph = tf.placeholder(tf.float32,
                                         [1, self.targetSize, self.targetSize, opt.IMG_SHAPES[2]],
                                         name='real_imgs')
            self.edge_ph = tf.placeholder(tf.float32, [self.targetSize, self.targetSize],
                                          name='edges')
            self.mask_ph = tf.placeholder(tf.float32, [self.targetSize, self.targetSize, 1],
                                          name='mask')

            self.G.build_test_model((self.img_ph, self.edge_ph), self.mask_ph, opt)

            with self.sk_sess.as_default():
                tf.global_variables_initializer().run()
                ckpt = tf.train.get_checkpoint_state(model_path)  # checkpoint
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                # restore
                vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    try:
                        var_value = tf.contrib.framework.load_variable(os.path.join(model_path, ckpt_name), from_name)
                        assign_ops.append(tf.assign(var, var_value))
                    except Exception:
                        continue
                self.sk_sess.run(assign_ops)

        # no Labels.
        print('SKI loaded.')

    @torch.no_grad()
    def forward(self, imgs, masks,**kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        if self.one_for_holes:
            masks = 1 - masks  # 1 for hole

        imgs = (imgs + 1.0) * 0.5
        real_imgs_np = tensor2cv(imgs, toRGB=False)
        mask_np = tensor2cv(masks, toRGB = True)
        comp_imgs = []

        with tf.device(f'/device:GPU:0'):
            for im, mask in zip(real_imgs_np, mask_np):
                img, edge = image_edge_processing(im)
                mask = (mask / 255.0).astype(np.float32)
                img = np.expand_dims(img, axis=0)
                with self.sk_graph.as_default():
                    raw_x, raw_x_incomplete, raw_x_complete = self.sk_sess.run(
                        [self.G.raw_x, self.G.raw_x_incomplete,
                         self.G.raw_x_complete],
                        feed_dict={self.img_ph: img, self.mask_ph: mask, self.edge_ph: edge})
                    comp_imgs.append(raw_x_complete.squeeze(axis=0))


            output = cv2tensor(comp_imgs).to(imgs.device)

            if output.size(-1) != w:
                output = F.interpolate(output, w, mode="bilinear")

        output = (output / 0.5) - 1.0
        return output







