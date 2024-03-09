import argparse
import time

from src.utils.util import find_model_using_name
from src.utils.util import get_file_info
from src.utils.im_process import get_transparent_mask
from tqdm import tqdm
from PIL import Image
import torch
import glob
import os

parse = argparse.ArgumentParser()
parse.add_argument('--device',type=str,dest='device',default="cuda",help='device')
parse.add_argument('--model_name',type=str,dest='model_name',default="MAT",help='the name of the model')
parse.add_argument('--model_path',type=str,dest='model_path',default="",help='the path of checkpoint')
parse.add_argument('--config_path',type=str,dest='config_path',default="",help='required for some of the models')
parse.add_argument('--sample_num',type=int,dest='sample_num',default=3,help='how many different output images for one input')
parse.add_argument('--target_size',type=int,dest='target_size',default=256,help='target image size for the model (not input size)')
parse.add_argument('--max_num',type=int,dest='max_num',default=100,help='the maximum number of testing images')
parse.add_argument('--edge_type',type=str,dest='edge_type',default="sobel",help='required for some of the models')
parse.add_argument('--img_path',type=str,dest='img_path',default="",help='image path for processing')
parse.add_argument('--mask_path',type=str,dest='mask_path',default="",help='mask path for processing')
parse.add_argument('--save_path',type=str,dest='save_path',default="./demo_results",help='path for saving the result')
parse.add_argument('--get_complexity', action='store_true',help='whether to calculate the image complexity')
arg = parse.parse_args()

def parse_path(path):
    if os.path.isdir(path):
        img_paths = sorted(glob.glob(path + "/*.png") + glob.glob(path + "/*.jpg"))
    elif os.path.isfile(path) and get_file_info(path)["file_type"] in ["png","jpg"]:
        img_paths = [path]
    else:
        raise Exception("Unrecognized image path!")

    return img_paths

def load_model(model_name, model_path, device, target_size, config_path = "", edge_type = "sobel"):
    print(f"Loading {model_name} from {model_path}...")
    package_name = "src.model_zoo." + model_name.lower()
    if config_path != "":
        model = find_model_using_name(package_name, model_name,
                                      model_path = model_path,
                                      config_path = config_path,
                                      device = device, targetSize = target_size,
                                      edge_type = edge_type)
    else:
        model = find_model_using_name(package_name, model_name,
                                      model_path=model_path,
                                      device=device, targetSize=target_size,
                                      edge_type=edge_type)
    return model

if __name__ == '__main__':
    inpaintingModel = load_model(
        model_name = arg.model_name,
        model_path = arg.model_path,
        device = arg.device,
        target_size = arg.target_size,
        edge_type = arg.edge_type
    )
    save_path = os.path.join(arg.save_path, arg.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_paths = parse_path(arg.img_path)
    mask_paths = parse_path(arg.mask_path)

    model_complex = {}

    print("Processing images...")

    time_cost = 0.0
    for j, (im_path, mask_path) in enumerate(tqdm(zip(img_paths, mask_paths))):
        file_name = get_file_info(im_path)["file_name"]
        img_t, gray_t, edge_t, mask_t = inpaintingModel.preprocess(im_path, mask_path)
        tmp_time_cost = 0.0
        for i in range(arg.sample_num):
            if hasattr(inpaintingModel, "state_dict_list"):
                inpaintingModel.G.load_state_dict(inpaintingModel.state_dict_list[i])

            start_time = time.time()
            out_t = inpaintingModel.forward(img_t, mask_t, grays = gray_t, edges = edge_t)
            tmp_time_cost += time.time() - start_time
            if arg.get_complexity and model_complex == {}:
                flops,param = inpaintingModel.get_complex()
                model_complex = {"Param": param, "FLOPs": flops}
            comp_img = img_t * mask_t + out_t * (1 - mask_t)
            comp_img_np = inpaintingModel.postprocess(comp_img)
            Image.fromarray(comp_img_np).save(save_path + f'/{arg.model_name}_{j:0>5d}_im_out_{i}.jpg')


        gt_img_np = inpaintingModel.postprocess(img_t)
        mask_np = (1 - mask_t[0]).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        masked_img_np, _ = get_transparent_mask(gt_img_np, mask_np)  # the input mask should be 1 for holes

        Image.fromarray(masked_img_np).save(save_path + f'/{arg.model_name}_{j:0>5d}_im_masked.jpg')
        Image.fromarray(gt_img_np).save(save_path + f'/{arg.model_name}_{j:0>5d}_im_truth.jpg')

        time_cost += tmp_time_cost / arg.sample_num
        if j >= arg.max_num:
            break

    print(model_complex)
    print(f"The inpainting results have been saved to {save_path}")
    print(f"Inference Speed: {time_cost * 1000 / len(img_paths)} ms")
    print("Done.")








