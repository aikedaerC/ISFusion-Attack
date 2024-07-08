import json 
import math
from math import exp
import torch.nn.functional as F
import numpy as np
import torch
import cv2
import os
from PIL import Image
from torchvision import transforms

class MSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(MSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    # Create a 1D Gaussian distribution vector
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    # Create a Gaussian kernel
    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)  # Add a dimension along axis 1
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # Create a 2D window
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    # Calculate SSIM using a normalized Gaussian kernel
    def mssim(self, img1, img2, window, window_size, channel=1, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        
        return self.mssim(img1, img2, self.window, self.window_size, channel, self.size_average)

def to_tensor(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    transform = transforms.ToTensor()
    obj_tensor = transform(img_pil)
    obj_tensor = obj_tensor.unsqueeze(0)
    return obj_tensor

def calculate_ssim(img1, img2, win_size=11):
    # import pdb;pdb.set_trace()
    img1 = to_tensor(img1)
    img2 = to_tensor(img2)
    mssim = MSSIM(channel=3)
    mssim_index = mssim(img1, img2)
    return mssim_index



def score(gt_json, px_json):
    with open(gt_json) as f:
        gt = json.load(f)

    with open(px_json) as f:
        px = json.load(f)

    # print(gt[0]['image'], gt[0]['color'], gt[0]['num'])

    asr = 0
    total = len(gt)
    for idx in range(total):
        one_asr = 0
        for k in gt[idx]['num'].keys():
            # num
            if gt[idx]['num'][k] != px[idx]['num'][k]:
                one_asr += 34 #0.5*((gt[idx]['num'][k] - px[idx]['num'][k])**2)
            # color
            if gt[0]['num'][k] != 0:
                if gt[idx]['color'][k] != px[idx]['color'][k]:
                    one_asr += 34
        asr += (one_asr)*(0.5+0.5*px[idx]['ssim'])
    avg_score = asr/total
    print(f"total score: {avg_score}")
    return avg_score


if __name__ == "__main__":

    # img1_p = "/home/data/images"
    # img2_p = "/home/data/p2/0.91_49/images"
    # sorted_files = os.listdir(img1_p)
    # for filename in sorted_files:
    #     img1_path = os.path.join(img1_p, filename)
    #     img2_path = os.path.join(img2_p, filename)
    #     im1 = cv2.imread(img1_path)  # im0.shape like (1028, 1912, 3)
    #     im2 = cv2.imread(img2_path)  # im0.shape like (1028, 1912, 3)
    #     ssim_score = calculate_ssim(im1, im2)
    #     print(ssim_score.item())
    #     break
    out_path = "/home/data/p2/0.935_41" #0.91_49"
    score("/home/data/labels_p2.json", os.path.join(out_path,"labels_p2.json"))