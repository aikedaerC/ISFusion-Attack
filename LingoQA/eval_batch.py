import subprocess

# Define the list of models
models = ["imp", "llava", "vila", "qwen"]

# Define the list of attack types
attack_types = ["sga_tcl"]# ["sga_albef", "sga_vitb16"] #["noise", "text", "noise_text"]

# Define the list of questions
questions = ["question2"] # "question1", 

# Loop through each combination of model, attack_type, and question 
for model in models:
    for attack_type in attack_types:
        for question in questions:
            # Construct the command
            command = f"python benchmark/evaluate.py --predictions_path /home/aikedaer/mydata/data/comp/predict/{attack_type}/{question}_{model}_predict.csv --answer_path /home/aikedaer/mydata/data/comp/{question}_{model}_answer.csv"
            
            # Execute the command
            try:
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Command executed successfully for {model}, {attack_type}, {question}")
                print(result.stdout.decode())  # Print the stdout of the command
            except subprocess.CalledProcessError as e:
                print(f"Error executing command for {model}, {attack_type}, {question}")
                print(e.stderr.decode())  # Print the error output if the command fails

