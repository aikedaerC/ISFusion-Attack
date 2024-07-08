import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
import json
from tqdm import tqdm
import math
import cv2
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
torch.set_default_device("cuda")


class MSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(MSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    # Create a 1D Gaussian distribution vector
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
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


def process_one_ssim(gt_img_pth, att_img_pth):
    im1 = cv2.imread(gt_img_pth)  # im0.shape like (1028, 1912, 3)
    im2 = cv2.imread(att_img_pth)  # im0.shape like (1028, 1912, 3)
    img1 = to_tensor(im1)
    img2 = to_tensor(im2)
    mssim = MSSIM(channel=3)
    mssim_index = mssim(img1, img2)
    return mssim_index.item()

def process_one_task(text,img_path,tokenizer):
    #Set inputs
    image = Image.open(img_path) # 000460
    # image.show()

    input_ids = tokenizer(text, return_tensors='pt').input_ids.to("cuda")
    image_tensor = model.image_preprocess(image).to("cuda")
    # import pdb;pdb.set_trace()
    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        images=image_tensor,
        use_cache=True)[0]
    answer = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    return answer

def parse_caption(caption):
    # Remove square brackets and split by commas
    caption = caption.strip('[]').split(', ')

    # Split each key-value pair and create a dictionary
    caption_dict = {pair.split(': ')[0]: pair.split(': ')[1] for pair in caption}
    return caption_dict

def batch_transfer(gt_img_path, img_path, out_path, js_name, tokenizer):
    os.makedirs(out_path, exist_ok=True)
    color_qs = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
          USER: <image>\nWhat are the colors of the cars, persons, motorcycles, traffic lights and road signals in the image respectively, only one color in each class? \
               Give me the answer in this format: [cars: color1, persons: color2, motorcycles: color3, traffic lights: color4, road signals: color5] ASSISTANT:"
    num_qs = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
          USER: <image>\nHow many cars, persons, motorcycles, traffic lights and road signals in the image respectively? \
              Give me the answer in this format: [cars: num1, persons: num2, motorcycles: num3, traffic lights: num4, road signals: num5] ASSISTANT:"   
    #"/content/images"
    json_data = []
    img_list = os.listdir(img_path)
    sorted_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
    # sorted_list = sorted_list[94:]
    count_except = 0
    idx = -1 # -1 as init
    while idx<len(sorted_list)-1:
        idx += 1
        print(f"Processing -------------------------------------------- {sorted_list[idx]} ------------------------------------------------")
        color_q = process_one_task(color_qs, os.path.join(img_path,sorted_list[idx]),tokenizer)
        num_q = process_one_task(num_qs, os.path.join(img_path,sorted_list[idx]),tokenizer)
        ssim = process_one_ssim(os.path.join(gt_img_path,sorted_list[idx]), os.path.join(img_path,sorted_list[idx]))
        # import pdb;pdb.set_trace()
        try:
            color_q = parse_caption(color_q)
            num_q = parse_caption(num_q)
        except: # [cars: red, white, blue, green, yellow, black,
            color_qs = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
                USER: <image>\nWhat are the colors of the cars, persons, motorcycles, traffic lights and road signals in the image respectively, only one color in each class? \
                 Give me the answer in this format: [cars: color1, persons: color2, motorcycles: color3, traffic lights: color4, road signals: color5] ASSISTANT:"
            count_except+=1
            if count_except < 11:
                print(f"try {count_except} times")
                idx = idx - 1
            continue
        color_map = {"red":0,"green":1,"blue":2,"black":3,"white":4,"yellow":5,"blonde":6,"purple":7,"brown":8,"tan":9, "pink":10}
        color = {k:color_map.get(v, -1) for k,v in color_q.items()}
        num = {k:int(v) for k,v in num_q.items()}
        js = {"image":sorted_list[idx], "color": color, "num": num, "ssim": ssim}
        print(f"json: {js}")
        json_data.append(js)

        if (idx % 50 == 0) or (idx==len(sorted_list)-1):
            json_string = json.dumps(json_data, indent=2)
            with open(os.path.join(out_path, js_name), 'w') as file:
                file.write(json_string)


if __name__ == "__main__":
    #Create model
    model = AutoModelForCausalLM.from_pretrained(
        "MILVLG/imp-v1-3b",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)

    import argparse

    parser = argparse.ArgumentParser(description='Example Argument Parser')
    parser.add_argument('--gpath', type=str, default='/home/data', help='path')
    parser.add_argument('--ipath', type=str, default='/home/data/p2/0.935_41', help='img_path')
    # parser.add_argument('--opath', type=str, default='/home/data/p2/0.935_41', help='out_path')

    args = parser.parse_args()

    json_name = "labels_p2.json"

    batch_transfer(os.path.join(args.gpath, "images"), os.path.join(args.ipath, "images"), args.ipath, json_name, tokenizer)

    # cal score
    from myutils.score import score
    avg_score = score(os.path.join(args.gpath, json_name), os.path.join(args.ipath, json_name))

    
    



