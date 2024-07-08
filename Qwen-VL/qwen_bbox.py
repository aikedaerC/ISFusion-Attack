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

pre_label_path = f"/home/aikedaer/mydata/data/comp/predict/{atype}/{qton}_qwen_{postr}.csv" 

# export CUDA_VISIBLE_DEVICES=3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
# torch.set_default_device("cuda")
default_device = torch.device("cuda:0")
torch.cuda.set_device(default_device)
torch.__version__


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# use bf16
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda:0", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

import pandas as pd
import os
from tqdm import tqdm 

def get_answer(img_path, question):
    file_pth = os.path.join(my_image_path,img_path)

    query = tokenizer.from_list_format([
        {'image': file_pth},
        {'text': question},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)

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

predictions

predictions.to_csv(pre_label_path)
