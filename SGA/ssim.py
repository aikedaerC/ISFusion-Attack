import os 
from skimage.metrics import structural_similarity
from tqdm import tqdm 
import cv2

def ssim(src,att):
    (score, diff) = structural_similarity(src, att, win_size=5, channel_axis=2, full=True)
    return score

image_dir = "/workspace/data/orimg/Images"
attac_dir = "/workspace/data/orimg/ALBEF/adv_result_p1/images"
SSIM = 0
img_count = 0
for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_count += 1
        img_path = os.path.join(image_dir, filename)
        att_path = os.path.join(attac_dir, filename)
        im0 = cv2.imread(img_path)  # im0.shape like (1028, 1912, 3)
        att = cv2.imread(att_path)
        SSIM += ssim(att, im0)
SSIM = SSIM/img_count

print(f"SSIM: {SSIM}")
with open("/workspace/SGA/ssim.py", 'a') as f:
    f.write(f"# SSIM: {SSIM}")# SSIM: 0.8218980499280227# SSIM: 1.0# SSIM: 0.8216141724404433# SSIM: 0.8210483459775794# SSIM: 0.5497368990403# SSIM: 0.7398025035907201# SSIM: 0.7394519103336026# SSIM: 0.8337717600114384# SSIM: 0.7515728282904615# SSIM: 0.7383367939975147# SSIM: 0.820085651252998