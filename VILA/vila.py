# my_label_path="/home/aikedaer/mydata/data/lingodt/val.csv"
# my_image_path="/home/aikedaer/mydata/data/lingodt"

# my_label_path="/home/aikedaer/mydata/data/comp/question1.csv"
# my_image_path="/home/aikedaer/mydata/data/comp/"
# qton = "question1"
# postr = "answer"

atype = "sga_tcl"
# atype = "text"
# atype = "noise_text"

qton = "question2"
postr = "predict" 

my_label_path="/home/aikedaer/mydata/data/comp/question2.csv"
my_image_path=f"/home/aikedaer/mydata/data/comp/attacked/phase2/{atype}"

pre_label_path = f"/home/aikedaer/mydata/data/comp/predict/{atype}/{qton}_vila_{postr}.csv" 


# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

import warnings
warnings.filterwarnings("ignore")
print(torch.cuda.device_count())

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file, args):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files, args):
    out = []
    for image_file in image_files:
        image = load_image(image_file, args)
        out.append(image)
    return out

def init_model(args):
    # Model
    disable_torch_init()

        
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base, device=args.device)

    return model, model_name, image_processor, tokenizer

def eval_model(args, model, model_name, image_processor, tokenizer):
    if args.video_file is None:
        image_files = image_parser(args)
        images = load_images(image_files, args)
    else:
        if args.video_file.startswith("http") or args.video_file.startswith("https"):
            print("downloading video from url", args.video_file)
            response = requests.get(args.video_file)
            video_file = BytesIO(response.content)
        else:
            assert osp.exists(args.video_file), "video file not found"
            video_file = args.video_file
        from llava.mm_utils import opencv_extract_frames
        images = opencv_extract_frames(video_file, args.num_video_frames)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("no <image> tag found in input. Automatically append one at the beginning of text.")
            # do not repeatively append the prompt.
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
    # print("input: ", qs)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

        
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # print(images_tensor.shape)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs



warnings.filterwarnings("ignore")
class MyArgs():
    def __init__(self, model_path, image_file, query, conv_mode, model_base=None, sep=",", temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512) -> None:
        self.model_path = model_path
        self.model_base = model_base
        self.image_file = image_file
        self.query = query
        self.conv_mode = conv_mode
        self.sep = sep
        self.temperature=temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens=max_new_tokens
        self.device = "cuda:0"

        self.video_file = None
        self.num_video_frames = None
    
args = MyArgs(model_path="/home/aikedaer/.cache/huggingface/hub/models--Efficient-Large-Model--Llama-3-VILA1.5-8b/snapshots/40f9b452727d7fd944f3f8a65274d87d76fc2676", image_file="demo_images/av.png", conv_mode="llama_3", query="<image>\n Please describe the traffic condition.", )
model, model_name, image_processor, tokenizer = init_model(args)

# asw = eval_model(args,model, model_name, image_processor, tokenizer)


import pandas as pd
import os
from tqdm import tqdm 
warnings.filterwarnings("ignore")

def get_answer(img_path, question):
    file_pth = os.path.join(my_image_path,img_path)
    args.query = "<image>\n " + question
    args.image_file = file_pth
    response = eval_model(args, model, model_name, image_processor, tokenizer)

    return response

def str2list(input_string):
    # Remove the brackets and split by newline character
    list_elements = input_string.strip("[]").split('\n')

    # Remove any extra spaces and quotes
    cleaned_list = [element.strip(" '") for element in list_elements]
    return cleaned_list

val = pd.read_csv(my_label_path)
val["images"] = val["images"].map(str2list)
predictions = pd.DataFrame(columns=["question_id", "segment_id", "answer"])

for index, row in tqdm(val.iterrows(), total=val.shape[0]):
    question_id = row['question_id']
    segment_id = row['segment_id']
    images = row['images']
    question = row['question']
    answer = row['answer']
    # only use the first frame
    try: 
        asw = get_answer(images[0], question)
    except Exception as e:
        asw = f"An exception occured: {e}"
    # import pdb;pdb.set_trace()
    # Create a DataFrame for the current row
    new_row = pd.DataFrame({
        "question_id": [question_id],
        "segment_id": [segment_id],
        "question": [question],
        "answer": [asw]
    })

    # Append the new row to the predictions DataFrame using pd.concat
    predictions = pd.concat([predictions, new_row], ignore_index=True)
    # break
predictions.to_csv(pre_label_path)
predictions


