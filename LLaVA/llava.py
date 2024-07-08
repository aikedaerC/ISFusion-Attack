# my_label_path="/home/aikedaer/mydata/data/lingodt/val.csv"
# my_image_path="/home/aikedaer/mydata/data/lingodt"

# my_label_path="/home/aikedaer/mydata/data/comp/question1.csv"
# my_image_path="/home/aikedaer/mydata/data/comp"
# qton = "question1"
# postr = "answer"

atype = "sga_tcl"
# atype = "text"
# atype = "noise_text"

qton = "question2"
postr = "predict"

my_label_path="/home/aikedaer/mydata/data/comp/question2.csv"
my_image_path=f"/home/aikedaer/mydata/data/comp/attacked/phase2/{atype}"

pre_label_path = f"/home/aikedaer/mydata/data/comp/predict/{atype}/{qton}_llava_{postr}.csv" 

from llava.model.builder import load_pretrained_model 
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
# export CUDA_VISIBLE_DEVICES=4
model_path = "liuhaotian/llava-v1.5-7b"
model_name=get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_4bit=True,
    device_map="cuda:0"
)


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
import re
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import torch

def image_parser(image_file):
    out = image_file.split(",")
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def chat(qs, image_file):
    # qs = "what color is the car?"
    # image_file = ""

    temperature = 0
    top_p = None
    num_beams = 1
    max_new_tokens = 100

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(image_file)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

import pandas as pd
import os
from tqdm import tqdm 

def get_answer(img_path, question):
    file_pth = os.path.join(my_image_path,img_path)

    response = chat(question, file_pth)

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
    asw = get_answer(images[0], question)
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

