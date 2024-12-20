import argparse
import os
import random
from src.clip import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity
from skimage import io, color

def ssim(src,att):
    # 计算结构相似性
    score = 0
    for idx in range(src.shape[0]):
        h1 = color.rgb2gray(src[idx].transpose((1, 2, 0)))
        h2 = color.rgb2gray(att[idx].transpose((1, 2, 0)))
        score += structural_similarity(h1, h2, data_range=1.0)
    score /= src.shape[0]
    return score

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

# Function to resize back to the original size
def resize_to_original(image_tensor, original_size):
    # Assuming image_tensor is a PyTorch tensor in C x H x W format
    return TF.resize(image_tensor, original_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        image = self.loader(path)
        original_size = image.size
        return original_tuple + (path, original_size)


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="ViT-B/32", type=str)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default="temp", type=str, help='the folder name of output')
    
    parser.add_argument("--cle_data_path", default=None, type=str, help='path of the clean images')
    parser.add_argument("--tgt_data_path", default=None, type=str, help='path of the target images')
    args = parser.parse_args()
    
    # load clip_model params
    alpha = args.alpha
    epsilon = args.epsilon
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)
     
    # ------------- pre-processing images/text ------------- #
    
    # preprocess images
    transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_res,args.input_res), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # torchvision.transforms.CenterCrop(args.input_res),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )
    clean_data    = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    target_data   = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)
    repeat_count = len(clean_data) // len(target_data) + 1
    # Create a new dataset by repeating target_data
    extended_target_data = ConcatDataset([target_data] * repeat_count)
    # Now trim the extended dataset to match the size of clean_data
    balanced_target_data = Subset(extended_target_data, range(len(clean_data)))
    
    data_loader_imagenet = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    data_loader_target   = torch.utils.data.DataLoader(balanced_target_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    clip_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(clip_model.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            torchvision.transforms.CenterCrop(clip_model.visual.input_resolution),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
    
    # CLIP imgs mean and std.
    inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])
    # import pdb;pdb.set_trace()
    # start attack
    args.num_samples = len(clean_data)
    for i, ((image_org, _, path, original_sizes), (image_tgt, _, pp, _)) in enumerate(zip(data_loader_imagenet, data_loader_target)):
        # import pdb;pdb.set_trace()
        # if args.batch_size * (i+1) > args.num_samples:
        #     break
        
        # (bs, c, h, w)
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)
        
        # get tgt featutres
        with torch.no_grad():
            tgt_image_features = clip_model.encode_image(clip_preprocess(image_tgt))
            tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True) # ([b,512])

        # -------- get adv image -------- #
        s_index = 1
        count = 0
        delta = torch.zeros_like(image_org, requires_grad=True)
        for j in range(args.steps):
        # while s_index > 0.95:
            count += 1
            adv_image = image_org + delta
            adv_image = clip_preprocess(adv_image)
            adv_image_features = clip_model.encode_image(adv_image)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True) # ([b, 512])

            embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))  # computed from normalized features (therefore it is cos sim.)
            embedding_sim.backward()
            
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d
            delta.grad.zero_()
            print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, s_index {s_index}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")
            # print(f"iter {i}/{args.num_samples//args.batch_size}, s_index {s_index}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")

            # adv_image = image_org + delta
            # adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
            # img_org = image_org/255.0
            # # import pdb;pdb.set_trace()
            # t1,t2 = adv_image.clone().cpu().detach().numpy(), img_org.clone().cpu().detach().numpy()
            # s_index = ssim(t1, t2)
            # if (count % 50 == 0):
            #     # save imgs
            #     adv_image = image_org + delta
            #     adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

            #     for path_idx in range(len(path)):
            #         folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            #         folder_to_save = os.path.join(args.output, folder)
            #         if not os.path.exists(folder_to_save):
            #             os.makedirs(folder_to_save, exist_ok=True)
            #         torchvision.utils.save_image(resize_to_original(adv_image[path_idx], (original_sizes[1][path_idx].item(),original_sizes[0][path_idx].item())), os.path.join(folder_to_save, name))

        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
        # import pdb;pdb.set_trace()
        for path_idx in range(len(path)):
            folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            folder_to_save = os.path.join(args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            torchvision.utils.save_image(resize_to_original(adv_image[path_idx], (original_sizes[1][path_idx].item(),original_sizes[0][path_idx].item())), os.path.join(folder_to_save, name))
