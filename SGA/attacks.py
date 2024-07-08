import torchattacks

import sys
import torch
import torch.nn as nn
from torchattacks import PGD
from torchvision import models, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = "cuda:0"

model = models.resnet18(pretrained=True).to(device).eval()

def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()

def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.savefig("./att.png")
    
def pdg_att(images, labels):
    
    atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    adv_images = atk(images, labels)
    return adv_images



def pdg_p(single_img, detect=False, single_label=None):
    if detect:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299,299), interpolation=transforms.InterpolationMode.BICUBIC),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
    ori_size = single_img.shape[:-1]
    if single_label is None:
        batch_label = torch.zeros(1, dtype=torch.long).to(device)
    else: 
        batch_label = single_label
    
    batch_img = preprocess(single_img).unsqueeze(0).to(device)

    adv_imgs = pdg_att(batch_img, batch_label)
    if detect:
        reverse_transform = transforms.Resize(ori_size, interpolation=transforms.InterpolationMode.BICUBIC)
        adv_patch = reverse_transform(adv_imgs.squeeze()).detach().cpu().numpy()
    else:
        adv_patch = adv_imgs.squeeze().detach().cpu().numpy()

    adv_patch = (np.transpose(adv_patch, (1, 2, 0)) * 255).astype(np.uint8)
    # cv2.imwrite("at.png", adv_patch)
    # pre = get_pred(model, adv_patch, device)
    return adv_patch


if __name__ == "__main__":
    # from PIL import Image

    img = cv2.imread("/workspace/test/000000.jpg")
    labels = torch.zeros(1, dtype=torch.long).to(device)
    print(f"labels:{labels}")

    ori_size = img.shape[:-1]
    reverse_transform = transforms.Resize(ori_size, interpolation=transforms.InterpolationMode.BICUBIC)

    img = preprocess(img).unsqueeze(0).to(device)

    adv_imgs = pdg_att(img,labels)
    adv_patch = reverse_transform(adv_imgs.squeeze()).detach().cpu().numpy()
    adv_patch = (np.transpose(adv_patch, (1, 2, 0)) * 255).astype(np.uint8)
    cv2.imwrite("at.png", adv_patch)

