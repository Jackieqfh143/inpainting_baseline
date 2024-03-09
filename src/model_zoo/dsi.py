from src.lib.DSI.net.vqvae import vq_encoder_spec, vq_decoder_spec
from src.lib.DSI.net.structure_generator import structure_condition_spec, structure_pixelcnn_spec
from src.lib.DSI.net.texture_generator import texture_generator_spec, texture_discriminator_spec
from src.utils.util import  tensor2cv,cv2tensor
import tensorflow as tf
import numpy as np
import cv2
import torch
from src.model_zoo.basemodel import BaseModel
import torch.nn.functional as F
tf.logging.set_verbosity(tf.logging.ERROR)


class DSI(BaseModel):

    def __init__(self,model_path, device, info = {},
                 targetSize=256, config_path = "./src/lib/DSI/configs/dsi.yaml", **kwargs):
        super(DSI, self).__init__(**kwargs)
        self.device_arg = device.replace('cuda', 'GPU')
        self.opt = self.load_config(config_path)
        self.info = info
        self.targetSize = targetSize
        self.one_for_holes = True
        self.device_arg = device.replace('cuda', 'GPU')

        self.dsi_graph = tf.Graph()
        with self.dsi_graph.as_default():
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sess_config.allow_soft_placement = True
            self.dsi_sess = tf.Session(config=sess_config, graph=self.dsi_graph)
            vq_encoder = tf.make_template('vq_encoder', vq_encoder_spec)
            vq_decoder = tf.make_template('vq_decoder', vq_decoder_spec)
            structure_condition = tf.make_template('structure_condition', structure_condition_spec)
            structure_pixelcnn = tf.make_template('structure_pixelcnn', structure_pixelcnn_spec)
            texture_generator = tf.make_template('texture_generator', texture_generator_spec)
            texture_discriminator = tf.make_template('texture_discriminator', texture_discriminator_spec)

            self.img_ph = tf.placeholder(tf.float32, shape=(1, self.targetSize, self.targetSize, 3))
            self.mask_ph = tf.placeholder(tf.float32, shape=(1, self.targetSize, self.targetSize, 1))
            self.e_sample = tf.placeholder(tf.float32, shape=(
            1, self.targetSize // 8, self.targetSize // 8, self.opt.embedding_dim))
            self.h_sample = tf.placeholder(tf.float32, shape=(
            1, self.targetSize // 8, self.targetSize // 8, 8 * self.opt.nr_channel_cond_s))

            top_shape = (self.targetSize // 8, self.targetSize // 8, 1)

            batch_pos = self.img_ph
            mask = self.mask_ph
            masked = batch_pos * (1. - mask)
            enc_gt = vq_encoder(batch_pos, is_training=False, **self.opt.vq_encoder_opt)
            dec_gt = vq_decoder(enc_gt['quant_t'], enc_gt['quant_b'], **self.opt.vq_decoder_opt)
            self.cond_masked = structure_condition(masked, mask, **self.opt.structure_condition_opt)
            pix_out = structure_pixelcnn(self.e_sample, self.h_sample, dropout_p=0., **self.opt.structure_pixelcnn_opt)
            pix_out = tf.reshape(pix_out, (-1, self.opt.num_embeddings))
            probs_out = tf.nn.log_softmax(pix_out, axis=-1)
            samples_out = tf.multinomial(probs_out, 1)
            samples_out = tf.reshape(samples_out, (-1,) + top_shape[:-1])
            self.new_e_gen = tf.nn.embedding_lookup(tf.transpose(enc_gt['embed_t'], [1, 0]), samples_out,
                                                    validate_indices=False)

            # Inpaint with generated structure feature maps
            self.gen_out = texture_generator(masked, mask, self.e_sample, **self.opt.texture_generator_opt)
            self.img_gen = self.gen_out * mask + masked * (1. - mask)

            # Discriminator
            # dis_out = texture_discriminator(tf.concat([self.img_gen, mask], axis=3),
            #                                 **self.opt.texture_discriminator_opt)

            # Restore full model
            restore_saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                restore_saver.restore(self.dsi_sess, ckpt.model_checkpoint_path)
                print('Full model restored ...')
            else:
                print('Restore full model failed! EXIT!')
                # sys.exit()

        # no Labels.
        print('DSI loaded.')


    def local_preprocess(self,imgs_in,masks_in):
        imgs = []
        masks = []
        for img_np,mask_np in zip(imgs_in,masks_in):
            img_np = cv2.resize(img_np, (self.targetSize,self.targetSize),
                                interpolation=cv2.INTER_LINEAR)
            # mask_np = np.expand_dims(mask_np, -1)

            # Normalize and reshape the image and mask
            img_np = img_np / 127.5 - 1.
            mask_np = mask_np / 255.
            img_np = np.expand_dims(img_np, 0)
            mask_np = np.expand_dims(mask_np, 0)
            imgs.append(img_np)
            masks.append(mask_np)
        return imgs,masks

    @torch.no_grad()
    def forward(self, imgs, masks,**kwargs):
        bt, c, h, w = imgs.size()
        if w != self.targetSize:
            imgs = F.interpolate(imgs, self.targetSize, mode="bilinear")
            masks = F.interpolate(masks, self.targetSize)

        if self.one_for_holes:
            masks = 1 - masks  # 1 for hole

        imgs = (imgs + 1.0) * 0.5
        imgs_np = tensor2cv(imgs, toRGB=False)
        mask_np = tensor2cv(masks, toRGB = True)

        imgs_, masks_ = self.local_preprocess(imgs_np, mask_np)
        comp_imgs = []
        for im, mask in zip(imgs_, masks_):
            with self.dsi_graph.as_default():
                with tf.device(f'/{self.device_arg}'):
                    cond_masked_np = self.dsi_sess.run(self.cond_masked, {self.img_ph: im, self.mask_ph: mask})
                feed_dict = {self.h_sample: cond_masked_np}
                e_gen = np.zeros(
                    (1, self.targetSize // 8, self.targetSize // 8, self.opt.embedding_dim),
                    dtype=np.float32)
                top_shape = (self.targetSize // 8, self.targetSize // 8, 1)
                for yi in range(top_shape[0]):
                    for xi in range(top_shape[1]):
                        feed_dict.update({self.e_sample: e_gen})
                        with tf.device(f'/{self.device_arg}'):
                            new_e_gen_np = self.dsi_sess.run(self.new_e_gen, feed_dict)
                        e_gen[:, yi, xi, :] = new_e_gen_np[:, yi, xi, :]
                with tf.device(f'/{self.device_arg}'):
                    img_gen_np = self.dsi_sess.run(self.img_gen,
                                                   {self.img_ph: im, self.mask_ph: mask, self.e_sample: e_gen})
                temp_out = ((img_gen_np[0] + 1.) * 127.5).astype(np.uint8)
                comp_imgs.append(temp_out)

        output = cv2tensor(comp_imgs).to(imgs.device)
        if output.size(-1) != w:
            output = F.interpolate(output, w, mode="bilinear")

        output = (output / 0.5) - 1.0
        return output


