import torch
from surrogates import (
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
    VisionTransformerFeatureExtractor,
)
from utils import get_list_image, save_list_images
from tqdm import tqdm
from attacks import SpectrumSimulationAttack, SSA_CommonWeakness
from torchvision import transforms
import os

images = get_list_image("/root/autodl-tmp/data/background_patch/background")
resizer = transforms.Resize((224, 224))
images = [resizer(i).unsqueeze(0) for i in images]

# import pdb;pdb.set_trace()
blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
# import pdb;pdb.set_trace()
clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
# import pdb;pdb.set_trace()
vit = VisionTransformerFeatureExtractor().eval().cuda().requires_grad_(False)
# import pdb;pdb.set_trace()
models = [vit, blip, clip]


def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max = ssa_N * num_models
    count = count % max
    count = count // ssa_N
    return count


ssa_cw_loss = EnsembleFeatureLoss(models, ssa_cw_count_to_index, feature_loss=torch.nn.MSELoss())


attacker = SSA_CommonWeakness(
    models,
    epsilon=16 / 255,
    step_size=1 / 255,
    total_step=100,
    criterion=ssa_cw_loss,
    ssim_threshold=0.3
)

dir =  "/root/autodl-tmp/data/attacked_background_patch_100/background" #"./attack_img_encoder_misdescription/"
os.makedirs(dir, exist_ok=True)
if not os.path.exists(dir):
    os.mkdir(dir)
id = 0
for i, x in enumerate(tqdm(images)):
    x = x.cuda()
    ssa_cw_loss.set_ground_truth(x)
    adv_x = attacker(x, None)
    save_list_images(adv_x, dir, begin_id=id)
    id += x.shape[0]
