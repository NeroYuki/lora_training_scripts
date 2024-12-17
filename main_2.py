from kohya_process import execute_training, execute_resize
import os
import json

# read dataprep_result.json
with open('dataprep_result.json', 'r') as f:
    prep_results = json.load(f)

for prep_result in prep_results:
    lora_path = execute_training(
        character=prep_result["character"],
        series=prep_result["series"],
        img_count=prep_result["img_count"],
        dataset_folder=prep_result["dataset"],
    )

    execute_resize(
        model_path=lora_path, 
        rank=32, 
        save_folder="D:/AIstuff/stable-diffusion-webui/models/Lora/sdxl_lora/auto_train/"
    )