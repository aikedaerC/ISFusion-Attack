def calculate_average(file_path):
    with open(file_path, 'r') as file:
        numbers = file.readlines()  # Read all lines from the file
        total_sum = 0
        count = 0
        for number in numbers:
            total_sum += float(number.strip())  # Convert string to float and add to total
            count += 1  # Increment count for each number read 

        if count == 0:
            return "No numbers to average."
        average = total_sum / count  # Calculate average
        return average

import os
# Specify the path to your file
root_path = "/home/aikedaer/mydata/LVLMs/bleurt/question_scores"
file_lst = os.listdir(root_path)
file_lst = [os.path.join(root_path, e) for e in file_lst if e.endswith("scores")]
for file_path in file_lst:
    average_value = calculate_average(file_path)
    print(f"The average of the numbers in the {file_path} is: {average_value}")
