from bert_score import score


# with open("/home/aikedaer/mydata/LVLMs/bert_score/example/hyps.txt") as f:
#     cands = [line.strip() for line in f]

# with open("/home/aikedaer/mydata/LVLMs/bert_score/example/refs.txt") as f:                            
#     refs = [line.strip() for line in f]
# P, R, F1 = score(cands, refs, lang='en', verbose=True)
import logging

logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='## %Y-%m-%d %H:%M:%S', level=logging.ERROR)

import pandas as pd

attack_type = ["sga_tcl"] #["sga_albef", "sga_vitb16"] #["noise", "text", "noise_text"]
model = ["imp", "llava", "vila", "qwen"]
phase = ['2'] #['1', '2']

for phase_item in phase:
    for attack_type_item in attack_type:
        for model_item in model:
            gt = f"/home/aikedaer/mydata/data/comp/question{phase_item}_{model_item}_answer.csv"

            predictions = f"/home/aikedaer/mydata/data/comp/predict/{attack_type_item}/question{phase_item}_{model_item}_predict.csv"

            df1 = pd.read_csv(gt)
            refs = list(df1['answer'].values)

            df2 = pd.read_csv(predictions)
            cands = list(df2['answer'].values)

            P, R, F1 = score(cands, refs, lang='en', verbose=True)
            logging.critical(f"phase_item: {phase_item}, attack_type_item: {attack_type_item}, model_item: {model_item}")
            logging.critical(f"mean persion score {P.mean()}, mean F1 score {F1.mean()}, mean R score {R.mean()}")