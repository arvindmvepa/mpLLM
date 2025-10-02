import torch
import pandas as pd
import os
import json


def get_full_attention_mask(inputs):
    return torch.ones(inputs.size()[:-1], device=inputs.device)


def process_textbook_fig_cap_dataset(base_dir='Nephrology/TextBook/pdffigures2/figure/data/',
                                     save_file='Nephrology/Textbook_data.csv'):
    data = []  # This list will store all the JSON data

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Check the depth of the current directory
        if root[len(base_dir):].count(os.sep) < 3:
            for filename in files:
                if filename.endswith('.json'):  # Check if the file is a JSON file
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r') as file:  # Open and read the file
                        content = json.load(file)  # Load the JSON data from the file
                        data.extend(content)  # Append the data to the list
        else:
            # Stop further walking into subdirectories beyond two levels
            dirs[:] = []

    # Convert the list of data to a DataFrame
    df = pd.DataFrame(data)
    df.to_csv(save_file, index=False)  # Save the data to a CSV file

