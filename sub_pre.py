import os
import json
from math import sqrt
from datetime import datetime

folder_path = 'data/hacka_doll_3'  # Replace with the actual folder path

# Function to calculate the average score
def calculate_avg_score(created_at, score):
    now = datetime.now()
    # assume iso time with milisecond with timezone, convert to utc
    time_diff = now - datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=None)
    return (score / sqrt(time_diff.total_seconds())) * 10_000

# List to store the extracted data
data = []

# Iterate over all JSON files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            created_at = json_data['danbooru']['created_at']
            score = json_data['danbooru']['score']
            avg_score = calculate_avg_score(created_at, score)
            data.append({'filename': filename, 'avg_score': avg_score})

# Sort the data in descending order based on avg_score
sorted_data = sorted(data, key=lambda x: x['avg_score'], reverse=True)

# Print the sorted data
for item in sorted_data:
    print(f"Filename: {item['filename']}, Avg Score: {item['avg_score']}")


# get top 15% of the files and its accompanying image file (.danbooru_<id>_meta.json and danbooru_<id>.<img_extension>), copy it to a new folder
top_15_percent = int(len(sorted_data) * 0.30)

# Create a new folder to store the top 15% of the files
new_folder_path = 'data/hacka_doll_3_top_15_percent'
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# Copy the top 15% of the files and its accompanying image file to the new folder
for i in range(top_15_percent):
    filename = sorted_data[i]['filename']
    # extract danbooru_<id> from .danbooru_<id>_meta.json and copy all file containing "danbooru_<id>"" to the new folder
    id = "danbooru_" + filename.split('_')[1]

    for file in os.listdir(folder_path):
        if id in file:
            file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(new_folder_path, file)
            with open(file_path, 'rb') as fsrc, open(new_file_path, 'wb') as fdst:
                fdst.write(fsrc.read())
            print(f"File {file} copied to {new_folder_path}")
    
    # Copy the .json file to the new folder
    json_file_path = os.path.join(folder_path, filename)
    new_json_file_path = os.path.join(new_folder_path, filename)
    with open(json_file_path, 'rb') as fsrc, open(new_json_file_path, 'wb') as fdst:
        fdst.write(fsrc.read())
    print(f"File {filename} copied to {new_folder_path}")

print(f"Top 15% of the files copied to {new_folder_path}")
