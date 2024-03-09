import os
import imageio
import glob
import random
import shutil
from src.utils.util import get_file_info
import uuid


def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)

    return

def prepare_imgs(img_dir, target_img_nums, save_dir, sample_num = 3):
    sample_num = sample_num + 2
    image_list = sorted(glob.glob(img_dir + "/*.png") + glob.glob(img_dir + "/*.jpg"))
    total_imgs_num = len(image_list) // sample_num
    selected_imgs_idx = []
    while len(selected_imgs_idx) < target_img_nums:
        rand_idx = random.randint(0, total_imgs_num - 1)
        if not rand_idx in selected_imgs_idx:
            selected_imgs_idx.append(rand_idx)

    selected_imgs = []
    for idx in selected_imgs_idx:
        start_idx = sample_num * (idx - 1)
        selected_imgs += image_list[start_idx: start_idx + sample_num]

    for im_path in selected_imgs:
        im_name = get_file_info(im_path)["file_full_name"]
        shutil.copy(im_path, save_dir + f"/{im_name}")


if __name__ == "__main__":
    img_dir = "results/Celeba-hq/thick_256/MobileFill"
    uuid_str = str(uuid.uuid4())[:6]
    save_dir = f"./gif_imgs_{uuid_str}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    target_img_nums = 100
    prepare_imgs(img_dir, target_img_nums, save_dir)

    selected_imgs = sorted(glob.glob(save_dir + "/*.png") + glob.glob(save_dir + "/*.jpg"))
    gif_name = f'face_{uuid_str}.gif'
    create_gif(selected_imgs, gif_name)