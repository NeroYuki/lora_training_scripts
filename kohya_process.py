import os
import subprocess, sys

kohya_path = "D:/AIstuff/kohya_ss"
def execute_training(character: str, series: str, img_count: int, dataset_folder: str):
    # execute ./venv/Scripts/activate to activate training environment
    # change working directory to kohya_path
    os.chdir(kohya_path)

    # create sampler prompt file 
    img_data_folder = kohya_path + "/dataset/auto_img"
    # clean up img_data_folder
    if os.path.exists(img_data_folder):
        # remove all symlink in img_data_folder
        for file in os.listdir(img_data_folder):
            os.unlink(img_data_folder + "/" + file)
    else:
        os.makedirs(img_data_folder)

    sampler_prompt_file = kohya_path + "/dataset/{character}_prompt.txt".format(character=character)

    with open(sampler_prompt_file, 'w') as f:
        f.write("1girl, {char}, {sr}, dress, night sky backdrop, masterpiece, best quality, innocent pose, whole body --n nsfw, worst quality, bad quality, jpeg artifact, bad anatomy, 2girls, cropped, watermark, artist signature, deformed body, upskirt view, more than 2 legs --w 768 --h 1280 --l 7 --s 25"
                .format(sr=series.replace('_', ' '), char=character.replace('_', ' ').replace('(', '\\(').replace(')', '\\)')))
        
    # create symlink between dataset_folder and img_data_folder
    os.symlink(dataset_folder, img_data_folder + "/1_{char}".format(char=character), target_is_directory=True)

    # calculate recommended epoch based on image count
    soft_step_limit = 1200

    if img_count > 300:
        # calculate epoch so that (epoch * img_count // 2) is less than soft_step_limit
        epoch = max((soft_step_limit * 2) // img_count, 1)
    elif img_count > 200:   #1200
        epoch = 6
    elif img_count > 100:   #900
        epoch = 9
    elif img_count > 60:    #720
        epoch = 12
    elif img_count > 40:    #600
        epoch = 14
    else:
        epoch = 15

    epoch_save = 1
    if epoch >= 10: epoch_save = 2 

    output_name = "na_" + character.replace('(', '').replace(')', '')

    p = subprocess.Popen(['venv/Scripts/accelerate', "launch", 
                            "--mixed_precision", "bf16",
                            "--num_cpu_threads_per_process", "2",
                            "sd-scripts/sdxl_train_network.py",
                            "--config_file", "lora_train_prodigy_poor",
                            "--train_data_dir", img_data_folder,
                            "--max_train_epochs", str(epoch),
                            "--sample_prompts", sampler_prompt_file,
                            "--save_every_n_epochs", str(epoch_save),
                            "--output_name", output_name], stdout=sys.stdout, stderr=sys.stderr)

    p.communicate()
    p.wait()

    return kohya_path + '/dataset/model/' + output_name + '.safetensors'

def execute_resize(model_path: str, rank: int = 16, save_folder = "D:/AIstuff/stable-diffusion-webui/models/Lora/sdxl_lora/auto_train/"):
    # execute ./venv/Scripts/activate to activate training environment
    # change working directory to kohya_path
    os.chdir(kohya_path)

    filename = model_path.split("/")[-1]

    # execute resize script
    p = subprocess.Popen(['venv/Scripts/python', "tools/resize_lora.py",
                            "--save_precision", "bf16",
                            "--new_rank", str(rank),
                            "--save_to", save_folder + filename.replace('.safetensors', '_r{rank}.safetensors'.format(rank=rank)),
                            "--model", model_path,
                            "--device", "cuda",
                            "--dynamic_method", "sv_fro",
                            "--dynamic_param", "0.9"], stdout=sys.stdout, stderr=sys.stderr)

    p.communicate()
    p.wait()