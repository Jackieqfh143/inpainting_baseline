from src.utils.visualize import show_selected_imgs
import argparse
from omegaconf import OmegaConf
import yaml
import os
import uuid
import random

parse = argparse.ArgumentParser()
parse.add_argument('--mode', type=str, dest='mode', default="random", help='random || fixed')
parse.add_argument('--experiment_name', type=str, dest='experiment_name', default="test", help='dataset name')
parse.add_argument('--model_names',type=str,dest='model_names',default="all",help='model_names can be either str "all" or single model name, also can accept name list')
parse.add_argument('--ids', type=str, dest='ids', default="",
                   help='required for mode == fixed, the idx of the images for showcasing')
parse.add_argument('--source_img_dir', type=str, dest='source_img_dir', default="", help='images dir for showcasing')
parse.add_argument('--save_dir', type=str, dest='save_dir', default="./results", help='path for saving the results')
parse.add_argument('--sample_num',type=int,dest='sample_num',default=3,help='how many different output images for one input')
parse.add_argument('--max_vis_num', type=int, dest='max_vis_num', default=200, help='the maximum number of images for showcasing ')
parse.add_argument('--im_size', type=int, dest='im_size', default=256, help='image size for showcasing ')
parse.add_argument('--total_num',type=int,dest='total_num',default=1000,help='total number of test images')
parse.add_argument('--max_im_row', type=int, dest='max_im_row', default=7, help='the number of images for each row')
arg = parse.parse_args()


if __name__ == '__main__':
    info_config_path = "./configs/paper_info.yaml"
    with open(info_config_path, 'r') as f:
        info_dict = OmegaConf.create(yaml.safe_load(f))

    source_dir = arg.source_img_dir
    model_list = []
    if arg.model_names == "all":
        dirs = os.listdir(source_dir)
        for d in dirs:
            if not os.path.isdir(source_dir+f'/{d}'):
                continue
            if "comparison" in d:
                continue
            model_list.append(d)
    else:
        model_list = arg.model_name.split(",")

    info_dict = {k:info_dict.get(k) for k in model_list}
    save_dir_name = "comparison_" + str(uuid.uuid4())[:6]
    save_path = os.path.join(arg.save_dir, save_dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    selected_ids = []
    if arg.mode == "random":
        while (len(selected_ids) < arg.max_vis_num):
            tmp = random.randint(0, arg.total_num - 1)
            if tmp not in selected_ids:
                selected_ids.append(tmp)
    else:
        if isinstance(arg.ids, str):
            if "," in arg.ids:
                selected_ids = arg.ids.split(",")
            else:
                selected_ids.append(int(arg.ids))

    show_selected_imgs(web_title=arg.experiment_name,
                       web_header=f"Inpainting result of {arg.experiment_name}",
                       web_dir=save_path,
                       source_dir=source_dir,
                       img_dir=save_path,
                       info_dict=info_dict,
                       im_size=arg.im_size,
                       sample_num=arg.sample_num,
                       max_im_row=arg.max_im_row,
                       selected_imgs_idx=selected_ids)
