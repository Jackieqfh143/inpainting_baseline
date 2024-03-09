import shutup
shutup.please()
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
from tqdm import tqdm
import argparse
import torch
import time
import os
import shutil
from src.evaluate.evaluation import validate
from src.data.dataset import Configurations, ValDataSet, DataLoaderX
from src.utils.im_process import *
from src.utils.util import save2csv
from src.utils.util import find_model_using_name
from omegaconf import OmegaConf
import yaml
import random




parse = argparse.ArgumentParser()
parse.add_argument('--device',type=str,dest='device',default="cuda",help='device')
parse.add_argument('--experiment_name',type=str,dest='experiment_name',default="test",help='the name of this experiment')
parse.add_argument('--dataset_name',type=str,dest='dataset_name',default="Place",help='dataset name')
parse.add_argument('--model_names',type=str,dest='model_names',default="all",help='model_names can be either str "all" or single model name, also can accept name list')
parse.add_argument('--update',type=str,dest='update',default="all",help='model names list which are needed to be updated')
parse.add_argument('--exclude_names',type=str,dest='exclude_names',default="",help='')
parse.add_argument('--mask_type',type=str,dest='mask_type',default="thick_256",help='the mask type')
parse.add_argument('--batch_size',type=int,dest='batch_size',default=20,help='batch size')
parse.add_argument('--target_size',type=int,dest='target_size',default=256,help='target image size')
parse.add_argument('--random_seed',type=int,dest='random_seed',default=2023,help='random seed')
parse.add_argument('--total_num',type=int,dest='total_num',default=10000,help='total number of test images')
parse.add_argument('--sample_num',type=int,dest='sample_num',default=3,help='how many different output images for one input')
parse.add_argument('--max_vis_num',type=int,dest='max_vis_num',default=200,help='maximum number of the visualized images')
parse.add_argument('--img_dir',type=str,dest='img_dir',default="",help='sample images for validation')
parse.add_argument('--mask_dir',type=str,dest='mask_dir',default="",help='sample masks for validation')
parse.add_argument('--save_dir',type=str,dest='save_dir',default="./results",help='path for saving the results')
parse.add_argument('--aspect_ratio_kept', action='store_true',help='keep the image aspect ratio when resize it')
parse.add_argument('--fixed_size', action='store_true',help='fixed the crop size')
parse.add_argument('--style_aug', action='store_true',help='fixed the crop size')
parse.add_argument('--center_crop', action='store_true',help='center crop')
parse.add_argument('--get_diversity', action='store_true',help='whether to calculate the image diversity')
arg = parse.parse_args()

def post_process(out,gt,mask,idx,save_path,save_name, sample_num):
    for i in range(gt.size(0)):
        gt_ = (gt[i] + 1.0) * 0.5
        gt_img_np = gt_.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        mask_np = (1 - mask[i]).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        masked_img_np, _ = get_transparent_mask(gt_img_np, mask_np)  #the input mask should be 1 for holes
        Image.fromarray(masked_img_np).save(save_path + f'/{save_name}_{i + idx :0>5d}_im_masked.jpg')
        Image.fromarray(gt_img_np).save(save_path + f'/{save_name}_{i + idx:0>5d}_im_truth.jpg')

        for j in range(sample_num):
            comp_im = out[j * gt.size()[0] + i] * (1 - mask[i]) + gt[i] * mask[i]
            comp_im = (comp_im + 1.0) * 0.5
            fake_img_np = comp_im.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            Image.fromarray(fake_img_np).save(save_path + f'/{save_name}_{i + idx:0>5d}_im_out_{j}.jpg')

def set_random_seed(random_seed=666,deterministic=False):
    if random_seed is not None:
        print("Set random seed as {}".format(random_seed))
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        if deterministic:
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            #for faster training
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

def load_configs(path = "./configs"):
    files = os.listdir(path)
    configs = Configurations()
    for file in files:
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            opt = OmegaConf.create(yaml.safe_load(f))

        if "checkpoints" in file:
            configs.checkpoints_info = opt
        else:
            configs.paper_info = opt

    return configs

def parse_str(conf, model_names, data_name):
    model_name_list = []
    if isinstance(model_names, str):
        if model_names == "all":
            pass
        else:
            model_name_list = model_names.strip().split(",")
    else:
        model_name_list = []

    checkpoint_info = conf.checkpoints_info
    if len(model_name_list) == 0:
        model_name_list = filter(lambda x: checkpoint_info.get(x).get(data_name) != None,
                                 list(conf.checkpoints_info.keys()))

    return model_name_list

def load_models(conf, model_names, update_names, data_name, device):
    checkpoints_info = conf.checkpoints_info
    model_name_list = parse_str(conf, model_names, data_name)
    update_name_list = parse_str(conf, update_names, data_name)
    target_model_name_list = list(set(model_name_list) & set(update_name_list))

    print("target_model_name_list: ", target_model_name_list)
    model_dict = {}
    for model_name in target_model_name_list:
        package_name = "src.model_zoo." + model_name.lower()
        temp_info = checkpoints_info.get(model_name)
        if data_name in temp_info.keys():
            model_path = temp_info.get(data_name)
        else:
            break

        target_size = temp_info.get("target_size")
        info = conf.paper_info.get(model_name)
        model = find_model_using_name(package_name, model_name,
                                      info = info, data_name = data_name, model_path = model_path,
                                      device = device, targetSize = target_size)

        if model != None:
            model_dict[model_name] = model

    return model_dict

if __name__ == '__main__':
    assert arg.max_vis_num < arg.total_num
    # experiment_name = arg.dataset_name + f"_{arg.mask_type}"
    # print("experiment_name: ", experiment_name)
    conf = load_configs()
    model_dict = load_models(conf, arg.model_names, arg.update, arg.dataset_name.lower(), arg.device)
    save_dir = os.path.join(arg.save_dir,arg.dataset_name, arg.experiment_name)
    save_path = os.path.join(save_dir,arg.mask_type)
    path_dict = {}
    for model_name in model_dict.keys():
        temp_save_path = os.path.join(save_path, model_name)
        if not os.path.exists(temp_save_path):
            os.makedirs(temp_save_path)
        else:
            shutil.rmtree(temp_save_path)
            os.makedirs(temp_save_path)

        path_dict[model_name] = temp_save_path

    set_random_seed(arg.random_seed)

    test_dataset = ValDataSet(arg.img_dir,arg.mask_dir,arg.total_num)

    test_dataloader = DataLoaderX(test_dataset,
                                 batch_size=arg.batch_size, shuffle=False, drop_last=False,
                                 num_workers=8,
                                 pin_memory=True)


    time_cost_dict = {model_name: 0.0 for model_name in model_dict.keys()}

    print("Processing images...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            batch_256, batch_512 = batch
            start_time = time.time()
            for model_name, model in model_dict.items():
                bts = batch_256 if model.targetSize == 256 else batch_512
                gt, gray, edge, mask = bts
                gt = gt.to(arg.device)
                gray = gray.to(arg.device)
                edge = edge.to(arg.device)
                mask = mask.to(arg.device)
                temp_save_path = path_dict.get(model_name)
                sample_num = 1 if not model.info.get("Pluralistic") else arg.sample_num
                out_imgs = []
                for j in range(sample_num):
                    if hasattr(model, "state_dict_list") and arg.style_aug:
                        model.G.load_state_dict(model.state_dict_list[j])
                    out = model.forward(gt, mask, grays = gray, edges = edge)
                    out_imgs.append(out)

                out_imgs = torch.cat(out_imgs,dim=0)
                post_process(out_imgs, gt, mask, i * arg.batch_size,
                             temp_save_path,model_name, sample_num)

                time_span = (time.time() - start_time) / (gt.size(0) * arg.sample_num)
                time_cost_dict[model_name] += time_span

    time_cost_dict = {k:(v * 1000) / (arg.total_num) for k,v in time_cost_dict.items()}

    result_dict = {}
    for model_name,model in model_dict.items():
        info = model.info
        temp_dict = {"Model": model_name}
        temp_save_path = path_dict.get(model_name)
        print(f"Validating Performance of {model_name}: \n")
        sample_num = 1 if not model.info.get("Pluralistic") else arg.sample_num
        scores_dict = {}
        for i in range(sample_num):
            tmp_scores_dict, *_ = validate(real_imgs_dir=temp_save_path,
                    comp_imgs_dir=temp_save_path,
                    device=arg.device,
                    get_FID=True,
                    get_LPIPS=True,
                    get_MLPIPS=True if model.info.get("Pluralistic") else False,
                    get_IDS=True,
                    real_suffix=["*_im_truth"],
                    fake_suffix=[f"*_im_out_{i}"])

            if scores_dict == {}:
                scores_dict = tmp_scores_dict
            else:
                scores_dict = {k:v + tmp_scores_dict.get(k) for k,v in scores_dict.items()}

        scores_dict = {k:v/sample_num for k,v in scores_dict.items()}
        temp_dict.update(scores_dict)
        temp_dict["Speed (ms/img)"] = time_cost_dict.get(model_name)
        temp_dict["Source"] = info.get("Source")
        temp_dict["#Param (M)"] = info.get(model.targetSize).get("Param")
        temp_dict["FLOPs (G)"] = info.get(model.targetSize).get("FLOPs")
        result_dict[model_name] = temp_dict

    update = True if len(arg.update) > 0 else False  #whether to update certain model , default update all the models
    save2csv(experiment_name = arg.experiment_name,
             save_results_dict = result_dict,
             saveDir=save_path,sortby='FID',update=update)






