
import subprocess

# Define the list of models
models = ["vila", "qwen", "imp", "llava"]

# Define the list of attack types
attack_types = ["sga_tcl"] #["sga_albef", "sga_vitb16"] #["noise", "text", "noise_text"]

# Define the list of questions
questions = ["question2"] #["question1", "question2"]

# Loop through each combination of model, attack_type, and question
for model in models:
    for attack_type in attack_types:
        for question in questions:
            command = f"python -m bleurt.score_files  /home/aikedaer/mydata/LVLMs/bleurt/references \
                -candidate_file=/home/aikedaer/mydata/LVLMs/bleurt/candidates/{attack_type}/{question}_{model}_predict \
                    -reference_file=/home/aikedaer/mydata/LVLMs/bleurt/references/{question}_{model}_answer \
                        -bleurt_checkpoint=BLEURT-20 \
                            -scores_file=/home/aikedaer/mydata/LVLMs/bleurt/question_scores/{question}_{model}_{attack_type}_scores"

            # Execute the command
            try:
                result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Command executed successfully for {model}, {attack_type}, {question}")
                print(result.stdout.decode())  # Print the stdout of the command
            except subprocess.CalledProcessError as e:
                print(f"Error executing command for {model}, {attack_type}, {question}")
                print(e.stderr.decode())  # Print the error output if the command fails
