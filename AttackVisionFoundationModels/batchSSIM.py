import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
import os
from tqdm import tqdm
import re
def extract_number(filename):
    s = re.findall('\d+', filename)
    return int(s[0]) if s else None

import torch
import torch.nn.functional as F
from math import exp

def load_image_as_tensor(image_array):
    # 确保图像数据为浮点型，范围为 [0.0, 1.0]
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32) / 255.0

    # 将图像数据转换为 PyTorch 张量
    image_tensor = torch.from_numpy(image_array)

    # 如果图像是灰度图，添加通道维度
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)  # (1, H, W)
    elif image_tensor.ndim == 3 and image_tensor.shape[2] == 3:  # 假设是 RGB
        image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)

    # 添加批量维度
    image_tensor = image_tensor.unsqueeze(0)  # (1, C, H, W)

    return image_tensor


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


class SSIMCalculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SSIM Calculator")
        self.geometry("800x600")

        # Setup image panels and buttons for individual image selection
        self.frame_images = tk.Frame(self)
        self.frame_images.pack(side=tk.TOP, pady=10)

        self.panel1 = tk.Label(self.frame_images, text="No Image Selected", borderwidth=2, relief="groove")
        self.panel1.pack(side=tk.LEFT, padx=10, pady=10)
        self.panel2 = tk.Label(self.frame_images, text="No Image Selected", borderwidth=2, relief="groove")
        self.panel2.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_load_image1 = tk.Button(self.frame_images, text="Load Image 1", command=lambda: self.load_image(1))
        self.btn_load_image1.pack(side=tk.LEFT, padx=10)
        self.btn_load_image2 = tk.Button(self.frame_images, text="Load Image 2", command=lambda: self.load_image(2))
        self.btn_load_image2.pack(side=tk.LEFT, padx=10)

        self.btn_calculate_ssim = tk.Button(self.frame_images, text="Calculate SSIM", command=self.calculate_individual_ssim)
        self.btn_calculate_ssim.pack(side=tk.LEFT, padx=10)

        # Directory selection and batch SSIM calculation
        self.btn_dir1 = tk.Button(self, text="Select Directory 1", command=lambda: self.load_directory(1))
        self.btn_dir1.pack(fill=tk.X, expand=True)
        self.btn_dir2 = tk.Button(self, text="Select Directory 2", command=lambda: self.load_directory(2))
        self.btn_dir2.pack(fill=tk.X, expand=True)

        self.btn_calculate_batch_ssim = tk.Button(self, text="Calculate Batch SSIM", command=self.calculate_batch_ssim)
        self.btn_calculate_batch_ssim.pack(fill=tk.X, expand=True)

        # Result label
        self.result_label = tk.Label(self, text="SSIM: Not Calculated", height=4)
        self.result_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.directory1 = None
        self.directory2 = None
        self.image1 = None
        self.image2 = None

    def load_image(self, img_no):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            image = io.imread(file_path)
            if image.ndim == 3:
                image = color.rgb2gray(image)
                image = (image * 255).astype(np.uint8)
            fixed_image = Image.fromarray(image).resize((250, 250), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(fixed_image)
            if img_no == 1:
                self.image1 = image
                self.panel1.config(image=photo, text="")
                self.panel1.image = photo
            else:
                self.image2 = image
                self.panel2.config(image=photo, text="")
                self.panel2.image = photo

    def calculate_individual_ssim(self):
        if self.image1 is not None and self.image2 is not None:
            mssim = MSSIM(channel=3)
            image1_tensor = load_image_as_tensor(self.image1)
            image2_tensor = load_image_as_tensor(self.image2)
            mssim_index = mssim(image1_tensor, image2_tensor)

            ssim_index = ssim(self.image1, self.image2, data_range=self.image1.max() - self.image1.min())
            self.result_label.config(text=f"SSIM: {ssim_index:.4f}\nM-SSIM: {mssim_index:.4f}")
        else:
            self.result_label.config(text="Please load both images.")

    def load_directory(self, dir_no):
        directory_path = filedialog.askdirectory()
        if directory_path:
            if dir_no == 1:
                self.directory1 = directory_path
            else:
                self.directory2 = directory_path
            messagebox.showinfo("Directory Loaded", f"Directory {dir_no} loaded successfully!")

    def calculate_batch_ssim(self):
        if not self.directory1 or not self.directory2:
            messagebox.showerror("Error", "Please select both directories.")
            return

        max_ssim = -1
        min_ssim = 2
        avg_ssim = 0
        max_pair = None
        min_pair = None
        images_f1 = os.listdir(self.directory1)
        images_f1 = [e for e in images_f1 if e.endswith(".jpg")]
        images_f1 = sorted(images_f1, key=extract_number)
        images_f2 = os.listdir(self.directory2)
        images_f2 = [e for e in images_f2 if e.endswith(".jpg")]
        images_f2 = sorted(images_f2, key=extract_number)
        images1 = [os.path.join(self.directory1, f) for f in images_f1 if f.endswith(('.png', '.jpg', '.jpeg'))]
        images2 = [os.path.join(self.directory2, f) for f in images_f2 if f.endswith(('.png', '.jpg', '.jpeg'))]
        # print(images1)
        # print(images2)
        for idx in tqdm(range(len(images1))):
            img1 = io.imread(images1[idx])
            # import pdb;pdb.set_trace()
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
            # if idx ==47:
            #     break
        avg_ssim /=len(images1)

        self.result_label.config(text=f"Max SSIM: {max_ssim:.4f} between {max_pair[0]} and {max_pair[1]}\nMin SSIM: {min_ssim:.4f} between {min_pair[0]} and {min_pair[1]}\nAvg SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    app = SSIMCalculator()
    app.mainloop()
