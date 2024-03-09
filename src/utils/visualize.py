import glob
import os
from .html import HTML
from .util import get_file_info
import re
import shutil


def model_sort(info_dict):
    key1 = lambda x: re.search(r"\d+", info_dict.get(x).get("Source")).group()
    sorted_model_names = sorted(info_dict.keys(), key=lambda x: (key1, x), reverse=False)

    temp1 = [x for x in sorted_model_names if not info_dict.get(x).get("Pluralistic")]
    temp2 = [x for x in sorted_model_names if info_dict.get(x).get("Pluralistic")]
    sorted_model_names = temp1 + temp2
    return sorted_model_names

def smart_add_imgs(total_ims, total_txts, total_links,
                   max_im_row, webpage, im_size):

    init_pos = 0
    end_pos = max_im_row
    while init_pos != end_pos:
        ims = total_ims[init_pos:end_pos]
        txts = total_txts[init_pos:end_pos]
        links = total_links[init_pos:end_pos]
        webpage.add_images(ims, txts, links, width=im_size)
        init_pos = end_pos
        end_pos = min(len(total_ims), init_pos + max_im_row)


def collect_imgs(selected_imgs_idx, sample_num,
                 source_dir, save_dir, info_dict, img_suffex = ".jpg"):
    print("Collecting images...")
    model_list = model_sort(info_dict)
    for idx in selected_imgs_idx:
        idx = int(idx)
        temp_save_img_dir = os.path.join(save_dir, str(idx))
        if not os.path.exists(temp_save_img_dir):
            os.makedirs(temp_save_img_dir)

        for model_name in model_list:
            temp_img_dir = os.path.join(source_dir, model_name)
            temp_sample_num = 1 if not info_dict.get(model_name).get("Pluralistic") else sample_num
            masked_name = f'{model_name}_{idx:0>5d}_im_masked{img_suffex}'
            txts = [masked_name]
            for i in range(temp_sample_num):
                comp_name = f'{model_name}_{idx:0>5d}_im_out_{i}{img_suffex}'
                txts.append(comp_name)
            gt_name = f'{model_name}_{idx:0>5d}_im_truth{img_suffex}'
            txts.append(gt_name)
            for i, n in enumerate(txts):
                im_path = os.path.abspath(os.path.join(temp_img_dir, n))
                if "truth" in n:
                    n = f"1_truth.jpg"
                elif "masked" in n:
                    n = f"0_masked.jpg"
                else:
                    n = "2_" + n

                new_im_path = os.path.join(temp_save_img_dir, n)
                shutil.copy(im_path, new_im_path)


def show_selected_imgs(web_title,web_header,web_dir, source_dir, img_dir,info_dict,selected_imgs_idx,
                      im_size=256, max_im_row = 9, sample_num = 3, img_suffex = ".jpg"):
    print("Inserting images to html...")
    webpage = HTML(web_dir, web_title)
    webpage.add_header(web_header)

    collect_imgs(selected_imgs_idx, sample_num, source_dir, img_dir,
                 info_dict, img_suffex)

    dirs = os.listdir(img_dir)
    dirs.sort()
    for d in dirs:
        if os.path.isdir(img_dir+f'/{d}'):
            total_ims, total_txts, total_links = [], [], []

            imgs_path = sorted(glob.glob(img_dir+f'/{d}/*.jpg'))
            im_dict = {}
            for im_path in imgs_path:
                im_full_name = get_file_info(im_path)['file_full_name']
                # tmp1 = re.findall(r"[A-Za-z]+", im_full_name)[0]
                tmp = im_full_name.split("_")
                if "masked.jpg" in tmp or "truth.jpg" in tmp:
                    tmp1 = tmp[1]
                else:
                    tmp1 = "_".join(tmp[1:-4])
                tmp2 = re.findall(r"\d+", im_full_name)
                if len(tmp2) > 1:
                    im_name = f"{tmp1}-{tmp2[-1]}"
                else:
                    im_name = tmp1.replace(".jpg","")
                im_dict[im_name] = f"{d}/{im_full_name}"


            temp_info_dict = {k:info_dict.get(k.split("-")[0]) for k in im_dict.keys() if k not in ["masked","truth"]}
            order_key_list = model_sort(temp_info_dict)
            total_ims += [im_dict["masked"], im_dict["truth"]]
            total_txts += ["\nInput\n", "\nGT\n"]
            total_links += [im_dict["masked"], im_dict["truth"]]
            for image_name in order_key_list:
                source = temp_info_dict.get(image_name).get("Source")
                image_path = im_dict[image_name]
                total_ims.append(image_path)
                total_txts.append(f"{image_name}\n{source}")
                total_links.append(image_path)

            init_pos = 0
            end_pos = max_im_row

            while init_pos != end_pos:
                ims = total_ims[init_pos:end_pos]
                txts = total_txts[init_pos:end_pos]
                links = total_links[init_pos:end_pos]
                webpage.add_images(ims, txts, links, width=im_size)
                init_pos = end_pos
                end_pos = min(len(total_ims),init_pos + max_im_row)

    webpage.save()




