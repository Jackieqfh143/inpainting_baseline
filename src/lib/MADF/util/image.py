import torch
from torchvision import transforms
import PIL


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x

def normalize(x,masks_in):
    imgs = []
    masks = []
    img_transform = transforms.Compose(
        [transforms.Resize(size=(256,256)), transforms.ToTensor(),
         transforms.Normalize(mean=MEAN, std=STD)])
    mask_transform = transforms.Compose(
        [transforms.Resize(size=(256,256), interpolation=PIL.Image.NEAREST), transforms.ToTensor()])

    for im,mask in zip(x,masks_in):
        im_t = img_transform(im)
        mask_t = mask_transform(mask)
        imgs.append(im_t)
        masks.append(mask_t)

    return torch.stack(imgs),torch.stack(masks)


