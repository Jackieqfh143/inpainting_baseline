import time
import os
import argparse
import glob
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import random

from inpaint_model import InpaintCAModel

#
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--flist', default='', type=str,
#     help='The filenames of image to be processed: input, mask, output.')
# parser.add_argument(
#     '--image_height', default=-1, type=int,
#     help='The height of images should be defined, otherwise batch mode is not'
#     ' supported.')
# parser.add_argument(
#     '--image_width', default=-1, type=int,
#     help='The width of images should be defined, otherwise batch mode is not'
#     ' supported.')
# parser.add_argument(
#     '--checkpoint_dir', default='', type=str,
#     help='The directory of GMCNN_tf checkpoint.')


if __name__ == "__main__":
    model_dir = '../../../checkpoints/DeepFill_v2/celeba-hq'
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, 256, 256*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            model_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')


    random.seed(66)
    imgs = glob.glob(os.path.join('/home/codeoops/CV/data/Celeba-hq/test_256','*.jpg'))
    masks = glob.glob(os.path.join('/home/codeoops/CV/data/mask/testing_mask_dataset','*.png'))
    total_imgs = len(imgs)
    total_masks = len(masks)
    count = 0
    total_time_span = 0.0
    for im_path in imgs:
        mask_index = random.randint(0,total_masks-1)
        save_path = './results/{}.jpg'.format(count)
        im = cv2.imread(im_path)
        mask = cv2.imread(masks[mask_index])
        image = cv2.resize(im, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 4
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        start_time = time.time()
        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        total_time_span += time.time() - start_time
        print('Processed: {}'.format(save_path))
        cv2.imwrite(save_path, result[0][:, :, ::-1])

        count += 1

    print('Time total: {:.2f}'.format(total_time_span))
    print('Inference speed: {:.2f}'.format((total_time_span)/total_imgs))
