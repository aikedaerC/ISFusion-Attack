# my_label_path="/home/aikedaer/mydata/data/lingodt/val.csv"
# my_image_path="/home/aikedaer/mydata/data/lingodt"

# my_label_path="/home/aikedaer/mydata/data/comp/question1.csv"
# my_image_path="/home/aikedaer/mydata/data/comp"
# qton = "question1"
# postr = "answer"
# to_path = my_label_path.replace(qton, qton+"_imp_"+postr)

# export CUDA_VISIBLE_DEVICES=1

atype = "sga_tcl"
# atype = "text"
# atype = "noise_text" noise
 
qton = "question2"
postr = "predict"

my_label_path="/home/aikedaer/mydata/data/comp/question2.csv"
my_image_path=f"/home/aikedaer/mydata/data/comp/attacked/phase2/{atype}"

pre_label_path = f"/home/aikedaer/mydata/data/comp/predict/{atype}/{qton}_imp_{postr}.csv" 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
# torch.set_default_device("cuda")
default_device = torch.device("cuda:0")
torch.cuda.set_device(default_device)
torch.__version__


#Create model
# import os
# os.environ['HF_ENDPOINT'] = 'hf-mirror.com'

model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True,
    # force_download=True,
    resume_download=True)
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)


import pandas as pd
import os
from tqdm import tqdm 

def get_answer(img_path, question):
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \
            USER: <image>\n{question} ASSISTANT:"
    file_pth = os.path.join(my_image_path,img_path)
    image0 = Image.open(file_pth) 
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(default_device)
    image_tensor0 = model.image_preprocess(image0).to(default_device)

    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=50,
        images=image_tensor0,
        use_cache=True)[0]
    asw = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    return asw
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