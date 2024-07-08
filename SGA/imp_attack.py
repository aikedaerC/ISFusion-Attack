import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
import json
torch.set_default_device("cuda")
import cv2
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random 
from ultralytics import YOLO
from torchvision import transforms
from transformers import BertForMaskedLM
from PIL import Image

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip
from attacks import pdg_p
import utils
import re
import ruamel.yaml as yaml
from skimage.metrics import structural_similarity
from attacker import SGAttacker, ImageAttacker, TextAttacker

from transformers import AutoModelForCausalLM, AutoTokenizer


def process_one_task(text,cv2img, model, tokenizer):
    #Set inputs
    # image = Image.open(img_path) # 000460
    image = Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB))
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

def ans2label(img, model, tokenizer):
    color_qs = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
          USER: <image>\nWhat are the colors of the cars, persons, motorcycles, traffic lights and road signals in the image respectively, only one color in each class?? \
               Give me the answer in this format: [cars: color1, persons: color2, motorcycles: color3, traffic lights: color4, road signals: color5] ASSISTANT:"
    num_qs = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
          USER: <image>\nHow many cars, persons, motorcycles, traffic lights and road signals in the image respectively? \
              Give me the answer in this format: [cars: num1, persons: num2, motorcycles: num3, traffic lights: num4, road signals: num5] ASSISTANT:"   
    try:
        color_q = process_one_task(color_qs,img, model, tokenizer)
        num_q = process_one_task(num_qs,img, model, tokenizer)
        color_q,num_q = parse_caption(color_q), parse_caption(num_q)
        color_map = {"red":0,"green":1,"blue":2,"black":3,"white":4,"yellow":5,"blonde":6,"purple":7,"brown":8,"tan":9}
        color = {k:color_map.get(v, -1) for k,v in color_q.items()} # the color assigned -1 must be false cause train set has no such color
        num = {k:int(v) for k,v in num_q.items()}
    except:
        color = {"cars": 1, "persons": 1, "motorcycles": 1, "traffic lights": 1, "road signals": 1}
        num = {"cars": 1, "persons": 1, "motorcycles": 1, "traffic lights": 1, "road signals": 1}
    return color,num

def compare(gt_color,gt_num, pred_color, pred_num):
    acc = 0
    for k in gt_color.keys():
        if gt_color[k]==pred_color[k]:
            acc += 1
        if gt_num[k]==pred_num[k]:
            acc += 1
    return acc

def convert(adv_img, reverse_trans, file_path=None):
    adv_patch = reverse_trans(adv_img.squeeze()).detach().cpu().numpy()
    adv_patch = (np.transpose(adv_patch, (1, 2, 0)) * 255).astype(np.uint8)
    # cv2.imwrite(file_path, adv_patch)
    return adv_patch


def load_model(model_name, model_ckpt, text_encoder, device, config):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)    
    if model_name in ['ALBEF', 'TCL']:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(model_ckpt, map_location='cpu')
    ### load checkpoint
    else:
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer
    
    try:
        state_dict = checkpoint['model']
    except:
        state_dict = checkpoint

    if model_name == 'TCL':
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    return model, ref_model, tokenizer

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

def get_dataset(img, annotation, transform,yolo_model, init=False, adv_img=None):
    img_name = os.path.basename(img)
    if init:
        cv2image = cv2.imread(img)

        ori_image = copy.deepcopy(cv2image)
        # R = copy.deepcopy(cv2image[:,:,2])
        # G = copy.deepcopy(cv2image[:,:,1])
        # B = copy.deepcopy(cv2image[:,:,0])
        # # 1. BGR -> GRB
        # cv2image[:,:,0] = G 
        # cv2image[:,:,1] = R 
        # cv2image[:,:,2] = B
        grb_image = copy.deepcopy(cv2image) 
    else:
        cv2image = adv_img
        ori_image = copy.deepcopy(adv_img)
        grb_image = None
    
    image = Image.fromarray(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
    ori_size = image.size
    image = transform(image)
    obj_patch = {"patch": [], "patch_size": []}
    results = yolo_model.predict(cv2image, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    topil = transforms.ToPILImage()
    if boxes is not None:
        for box in boxes: 
            p = cv2image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            patch_tensor = transform(topil(p))
            obj_patch["patch"].append(patch_tensor)
            obj_patch["patch_size"].append(p.shape[:-1])
    
    texts = []
    for cpon in annotation[img_name]:
        texts.append(pre_caption(cpon, max_words=30))

    return image, grb_image, ori_image, texts, ori_size, boxes, obj_patch


def attacks(model, ref_model, dataset, tokenizer, device):
    # test
    model.float()
    model.eval()
    ref_model.eval()

    print('Computing features for evaluation adv...')
    # 132.5366991	139.9011609	148.1239298	44.71878048	46.11609075	47.00605992
    # 0.519751761	0.548632003	0.580878156	0.175367767	0.180847415	0.18433749

    images_normalize = transforms.Normalize((0.519751761, 0.548632003, 0.580878156), (0.175367767, 0.180847415, 0.18433749))
    img_attacker = ImageAttacker(images_normalize, eps=2/255, steps=20, step_size=0.5/255)
    txt_attacker = TextAttacker(ref_model, tokenizer, cls=False, max_length=30, number_perturbation=1,
                                topk=10, threshold_pred_score=0.3)
    attacker = SGAttacker(model, img_attacker, txt_attacker)

    scales = [0.5,0.75,1.25,1.5]

    mode = {"whole_image": False, "patch": True}
    image, bgr_image, ori_image, texts, ori_size, boxes, obj_patch = dataset
    txt2img = [0,0,0]
    if mode["whole_image"]:
        image = image.unsqueeze_(0).to(device)  
        # import pdb;pdb.set_trace()                                                                
        adv_images, adv_texts = attacker.attack(image, texts, txt2img, device=device,
                                                max_lemgth=30, scales=scales) 
        reverse_transform = transforms.Resize((ori_size[1],ori_size[0]), interpolation=transforms.InterpolationMode.BICUBIC)
        adv_whole_image = convert(adv_img=adv_images, reverse_trans=reverse_transform)
    if mode["patch"]:
        if boxes is not None:
            for idx, (patch, patch_size) in enumerate(zip(obj_patch["patch"], obj_patch["patch_size"])):
                # import pdb;pdb.set_trace()
                # 1.0 sga attack  
                # patch = patch.unsqueeze_(0).to(device)                                                                
                # adv_images, adv_texts = attacker.attack(patch, texts, txt2img, device=device,
                #                                         max_lemgth=30, scales=scales) 
                # reverse_transform = transforms.Resize((patch_size), interpolation=transforms.InterpolationMode.BICUBIC)
                # adv_patch = reverse_transform(adv_images.squeeze()).detach().cpu().numpy()
                # adv_patch = (np.transpose(adv_patch, (1, 2, 0)) * 255).astype(np.uint8)
                # 1.1 dpg attack
                box = boxes[idx]
                patch_ori = ori_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                adv_patch = pdg_p(patch_ori, detect=True)
                
                # 2. pin patch to ori image
                ori_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = adv_patch
        adv_whole_image = ori_image
    return adv_whole_image
            
def write_acc(outfile_path, filename, acc,ssim_score, total_score=-1, promote=0):
    with open(outfile_path, "a") as f:
        f.write(f"\n{filename},{acc},{ssim_score},{total_score},{promote}")

def ssim(src,att):
    (score, diff) = structural_similarity(src, att, win_size=5, channel_axis=2, full=True)
    return score

def main(source_model, source_ckpt, source_text_encoder, config_yaml,ann_file,root_path,output_dir):
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    ### MOdel ###
    print("Creating Source Model")
    # train model
    # source_model = "ViT-B/16" #  ALBEF
    # source_ckpt = "./checkpoint/albef_retrieval_flickr.pth" # used for ALBEF
    # source_text_encoder = "bert-base-uncased"
    # config_yaml = "/workspace/SGA/configs/Retrieval_flickr.yaml"
    config = yaml.load(open(config_yaml, 'r'), Loader=yaml.Loader)
    src_model, ref_model, src_tokenizer = load_model(source_model, source_ckpt, source_text_encoder, device, config)
    # test model
    # tgt_model = AutoModelForCausalLM.from_pretrained(
    #     "MILVLG/imp-v1-3b",
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     trust_remote_code=True)
    # tgt_tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)
    yolo_model = YOLO("yolov8n.pt")

    #### Dataset ####
    print("Creating dataset")
    # ann_file = '/workspace/data/orimg/captions.json'
    with open(ann_file, 'r') as f:
        ann = json.load(f)
    annotation = {}
    for x in ann["captions"]:
        if x["image"] in annotation.keys():
            annotation[x["image"]] = annotation[x["image"]] + " " + x["caption"]
        else:
            annotation[x["image"]] = x["caption"]

    n_px = config['image_res'] # src_model.visual.input_resolution
    s_test_transform = transforms.Compose([
        transforms.Resize((n_px,n_px), interpolation=Image.BICUBIC),
        transforms.ToTensor(),       
    ])
    

    # read gt color and num
    # imp_labels = '/root/labels.json'
    # with open(imp_labels) as f:
    #     gt = json.load(f)

    # root_path = "/workspace/data/orimg/Images"
    # output_dir = "/workspace/data/orimg/adv_result_p1/images/"
    os.makedirs(output_dir, exist_ok=True)
    acc_path = output_dir.replace("images","scores")
    os.makedirs(acc_path, exist_ok=True)

    
    for iname in annotation.keys():
        for i in range(4):
            img = os.path.join(root_path, iname)
            if i == 0: 
                adv_img = img
                ifo = True
            dataset = get_dataset(img, annotation, s_test_transform,yolo_model,init=ifo, adv_img=adv_img)
            ifo = False
            adv_img = attacks(src_model, ref_model, dataset, src_tokenizer, device)
        cv2.imwrite(os.path.join(output_dir, iname), adv_img)


    # idx = start_idx #299 #-1
    # good_score = True
    # adv_img = None
    # while idx < len(gt)-1:
    #     idx+=1
    #     print(f"\nProcessing ----------------------------{gt[idx]['image']}----------------------------------")
    #     img = os.path.join(root_path, gt[idx]["image"])
    #     dataset = get_dataset(img, annotation, s_test_transform, yolo_model, init=good_score, adv_img=adv_img)
    #     if good_score:
    #         adv_img = None
    #         tolerance_count = 0
    #         src_img = dataset[2]
            # grb image
            # pred_color, pred_num = ans2label(dataset[1], tgt_model, tgt_tokenizer) 
            # ssim_score = ssim(src_img, dataset[1])
            # accuracy = compare(gt[idx]["color"],gt[idx]["num"], pred_color, pred_num)
            # base_score = (11-accuracy) * ssim_score
            # max_acc = base_score
            # max_acc_adv_img = {"accuracy":accuracy, "ssim_score": ssim_score, "total_score": base_score, "adv": dataset[1]}

        # adv_img = attacks(src_model, ref_model, dataset, src_tokenizer, device)
        # ssim_score = ssim(src_img, adv_img)
        # pred_color, pred_num = ans2label(adv_img, tgt_model, tgt_tokenizer) # {'cars': blue, }
        # accuracy = compare(gt[idx]["color"],gt[idx]["num"], pred_color, pred_num)
        # least_acc = (11-accuracy) * ssim_score        
        # if accuracy < least_acc:
        #     least_acc = accuracy        
        # total_score = (11-accuracy) * ssim_score
        # if total_score > max_acc:
        #     max_acc = total_score
        #     max_acc_adv_img.update({
        #         "accuracy":accuracy,
        #         "ssim_score": ssim_score,
        #         "total_score": total_score,
        #         "adv": adv_img
        #     })

        # print(f"\n-----------------------Answer accuracy is {accuracy}; SSIM Score is {ssim_score}; TOTAL score is {total_score} ---------------------------")
        # if total_score < avg_acc:
        #     if tolerance_count > to :
        #         cv2.imwrite(os.path.join(output_dir, gt[idx]["image"]), max_acc_adv_img["adv"])
        #         write_acc(os.path.join(acc_path, score_filename), gt[idx]["image"],max_acc_adv_img["accuracy"],max_acc_adv_img["ssim_score"], total_score=max_acc_adv_img["total_score"])
        #         good_score = True
        #     else:
        #         good_score = False
        #         idx-=1
        #         tolerance_count += 1
        # else:
        #     cv2.imwrite(os.path.join(output_dir, gt[idx]["image"]), adv_img)
        #     write_acc(os.path.join(acc_path, score_filename), gt[idx]["image"], accuracy,ssim_score, total_score=total_score, promote=total_score-base_score)
        #     good_score = True

    

if __name__ == "__main__":
    root_path = "/home/aikedaer/mydata/data/comp/images/phase2"
    output_dir = "/home/aikedaer/mydata/data/comp/attacked/phase2/sga_tcl"
    source_model = "TCL" # "ViT-B/16" #  ALBEF
    source_ckpt = "./checkpoint/tcl_retrieval_flickr.pth" # albef_retrieval_flickr used for ALBEF
    source_text_encoder = "bert-base-uncased"
    config_yaml = "/home/aikedaer/mydata/LVLMs/SGA/configs/Retrieval_flickr.yaml"
    ann_file = '/home/aikedaer/mydata/data/comp/question2_captions.json'
    main(source_model, source_ckpt, source_text_encoder, config_yaml,ann_file,root_path,output_dir)
            





