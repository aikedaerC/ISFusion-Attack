# #!/bin/bash

# certain hyper-parameters can be modified based on user's preference

# python _train_adv_img_query_minigpt4.py  \
#     --batch_size 10 \
#     --num_samples 10000 \
#     --steps 100 \
#     --output "minigpt4_adv" \
import os


cle_data_path = os.path.join("/home/aikedaer/mydata/data/comp/images/phase2_patch/background_patch")
tgt_data_path = os.path.join("/home/aikedaer/mydata/data/comp/attack_targets")
out_dir = os.path.join("/home/aikedaer/mydata/data/comp/attacked/phase2/unidiff_minigpt4")
os.makedirs(out_dir, exist_ok=True)

exc = f"python _train_adv_img_trans.py --output {out_dir} --epsilon 12  --batch_size 10  --steps 100 --cle_data_path {cle_data_path} --tgt_data_path {tgt_data_path}"
os.system(exc)


# cle_data_path = os.path.join("/root/autodl-tmp/data/background_patch") 
# tgt_data_path = os.path.join("/root/autodl-tmp/data/tgt_icon")
# out_dir = os.path.join("/root/autodl-tmp/data/attacked_background_patch")
# os.makedirs(out_dir, exist_ok=True)

# exc = f"python _train_adv_img_trans.py --output {out_dir} --epsilon 12  --batch_size 10  --steps 100 --cle_data_path {cle_data_path} --tgt_data_path {tgt_data_path}"
# os.system(exc)