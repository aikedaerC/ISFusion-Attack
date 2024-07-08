import argparse
import os
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
from math import exp
import numpy as np

def extract_number(filename):
    s = re.findall('\d+', filename)
    return int(s[0]) if s else None

def load_image_as_tensor(image_array):
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim == 3 and image_tensor.shape[2] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

class MSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(MSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

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

def calculate_individual_ssim(image1_path, image2_path):
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)
    if image1.ndim == 3:
        image1 = color.rgb2gray(image1)
    if image2.ndim == 3:
        image2 = color.rgb2gray(image2)
    ssim_index = ssim(image1, image2, data_range=image1.max() - image1.min())
    mssim = MSSIM(channel=1)
    image1_tensor = load_image_as_tensor(image1)
    image2_tensor = load_image_as_tensor(image2)
    mssim_index = mssim(image1_tensor, image2_tensor).item()
    print(f"SSIM: {ssim_index:.4f}\nM-SSIM: {mssim_index:.4f}")

def calculate_batch_ssim(directory1, directory2):
    max_ssim = -1
    min_ssim = 2
    avg_ssim = 0
    max_pair = None
    min_pair = None
    images_f1 = os.listdir(directory1)
    images_f1 = [e for e in images_f1 if e.endswith(".jpg")]
    images_f1 = sorted(images_f1, key=extract_number)
    images_f2 = os.listdir(directory2)
    images_f2 = [e for e in images_f2 if e.endswith(".jpg")]
    images_f2 = sorted(images_f2, key=extract_number)
    images1 = [os.path.join(directory1, f) for f in images_f1 if f.endswith(('.png', '.jpg', '.jpeg'))]
    images2 = [os.path.join(directory2, f) for f in images_f2 if f.endswith(('.png', '.jpg', '.jpeg'))]

    for idx in tqdm(range(len(images1))):
        img1 = io.imread(images1[idx])
        img1 = color.rgb2gray(img1)
        img2 = io.imread(images2[idx])
        img2 = color.rgb2gray(img2)
        index = ssim(img1, img2, data_range=1.0)
        avg_ssim += index
        if index > max_ssim:
            max_ssim = index
            max_pair = (os.path.basename(images1[idx]), os.path.basename(images2[idx]))
        if index < min_ssim:
            min_ssim = index
            min_pair = (os.path.basename(images1[idx]), os.path.basename(images2[idx]))
    avg_ssim /= len(images1)
    print(f"Max SSIM: {max_ssim:.4f} between {max_pair[0]} and {max_pair[1]}")
    print(f"Min SSIM: {min_ssim:.4f} between {min_pair[0]} and {min_pair[1]}")
    print(f"Avg SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SSIM for images")
    parser.add_argument("mode", choices=["single", "batch"], help="Mode of operation: single or batch") 
    parser.add_argument("input1", help="Path to the first image or directory")
    parser.add_argument("input2", help="Path to the second image or directory")
    args = parser.parse_args()

    if args.mode == "single":
        calculate_individual_ssim(args.input1, args.input2)
    elif args.mode == "batch":
        calculate_batch_ssim(args.input1, args.input2)
