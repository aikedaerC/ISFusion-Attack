import torch
from PIL import Image
import numpy as np
from torch import Tensor


@torch.no_grad()
def save_image(x: Tensor, path="./0.png") -> Image.Image:
    if len(x.shape) == 4:
        x = x.squeeze(0)
    x = x.permute(1, 2, 0) * 255
    x = x.cpu().numpy()
    if x.shape[2] == 1:
        # cv2.imwrite(path, x.squeeze())
        return x.squeeze()
    x = Image.fromarray(np.uint8(x))
    # x.save(path)
    return x



def attack_img_encoder(x_tensor: Tensor, ssa_cw_loss, attacker) -> Image.Image:
    x = x_tensor.cuda()
    ssa_cw_loss.set_ground_truth(x)
    # import pdb;pdb.set_trace()
    adv_x = attacker(x, None)
    
    return save_image(adv_x)
