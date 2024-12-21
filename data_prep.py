import os
from PIL import Image
import re
import json
from math import sqrt, pow
from datetime import datetime
import random
import questionary
from tqdm import tqdm
from waifuc_process import local_process, remote_crawl, local_process_raw
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch import torch

# Function to calculate the average score
# def calculate_avg_score(created_at, score):
#     now = datetime.now()
#     # assume iso time with milisecond with timezone, convert to utc
#     time_diff = now - datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=None)
#     return (score / sqrt(time_diff.total_seconds())) * 10_000

# additional tagging based on danbooru metadata
def assign_extra_tags(meta_folder_path, folder_path, beginning_tag):
    # data = []
    # Iterate over all JSON files in the folder
    for filename in tqdm(os.listdir(meta_folder_path)):
        if filename.endswith('.json'):
            combined_tags = beginning_tag
            combined_meta_tags = ''
            general_tag = []
            char_tag = []
            series_tag = []
            artist_tag = []

            file_path = os.path.join(meta_folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                if 'yande' in json_data:
                    # time in unix timestamp
                    created_at = datetime.fromtimestamp(json_data['yande']['created_at']).replace(tzinfo=None)
                    rating = json_data['yande']['rating']
                elif 'gelbooru' in json_data:
                    # timing format is different between danbooru and gelbooru (Thu Nov 11 19:30:14 -0600 2021)
                    created_at = datetime.strptime(json_data['gelbooru']['created_at'], "%a %b %d %H:%M:%S %z %Y").replace(tzinfo=None)
                    rating = json_data['gelbooru']['rating']
                elif 'pixiv' in json_data:
                    created_at = datetime.strptime(json_data['pixiv']['create_date'], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
                    rating = 'e' if json_data['pixiv']['x_restrict'] == 1 else 'g'
                else:
                    created_at = datetime.strptime(json_data['danbooru']['created_at'], "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=None)
                    rating = json_data['danbooru']['rating']
                    general_tag = json_data['danbooru']['tag_string_general'].split(' ')
                    char_tag = json_data['danbooru']['tag_string_character'].split(' ')
                    series_tag = json_data['danbooru']['tag_string_copyright'].split(' ')
                    artist_tag = json_data['danbooru']['tag_string_artist'].split(' ')
                # rating map
                # safe	            g
                # sensitive	    s
                # nsfw	            q
                # explicit, nsfw	    e

                # created at map
                # Year Tag	        Year Range
                # newest	        2021 to 2024
                # recent	        2018 to 2020
                # mid	        2015 to 2017
                # early	        2011 to 2014
                # oldest	        2005 to 2010

                # rating tag
                if rating == "s" or rating == "sensitive":
                    combined_meta_tags += ", sensitive"
                elif rating == "q" or rating == "questionable":
                    combined_meta_tags += ", nsfw"
                elif rating == "e" or rating == "explicit":
                    combined_meta_tags += ", explicit"

                # year tag
                if created_at.year >= 2021:
                    combined_meta_tags += ", newest"
                elif created_at.year >= 2018:
                    combined_meta_tags += ", recent"
                elif created_at.year >= 2015:
                    combined_meta_tags += ", mid"
                elif created_at.year >= 2011:
                    combined_meta_tags += ", early"
                elif created_at.year >= 2005:
                    combined_meta_tags += ", oldest"
                else:
                    combined_meta_tags += ", ancient"

                combined_tags = combined_tags + ((', artist: ' + ', '.join(artist_tag)) if len(artist_tag) > 0 else '')  + combined_meta_tags

            # search for file sharing the same id in folder path as the meta file
            file_comp = filename.split('_')
            # file_comp len should be 3 minimum
            id = "danbooru_" + '_'.join(file_comp[1:(max(len(file_comp) - 1, 2))])
            gel = "gelbooru_" + '_'.join(file_comp[1:(max(len(file_comp) - 1, 2))])
            yan = "yande_" + '_'.join(file_comp[1:(max(len(file_comp) - 1, 2))])
            pix = "pixiv_" + '_'.join(file_comp[1:(max(len(file_comp) - 1, 2))])
            for file in os.listdir(folder_path):
                if (id == file.split('.')[0]) and file.endswith(".txt"):
                    # original file, use original tagging if applicable
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, "r+") as file:
                        content = file.read()
                        content = general_tag[0] + ", " + ', '.join(char_tag) + ', ' + ', '.join(series_tag) + ', artist: ' + ', '.join(artist_tag) + combined_meta_tags + ', ' + ', '.join(general_tag[1:])
                        content = content.replace("_", " ")
                        file.seek(0)
                        file.write(content)
                        file.truncate()
                elif (id in file or gel in file or yan in file or pix in file) and file.endswith(".txt"):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, "r+") as file:
                        match_pattern = re.compile(r"1(girl|boy), ")
                        content = file.read()
                        content = match_pattern.sub("", content)
                        content = combined_tags + ", " + content
                        content = content.replace("_", " ")
                        file.seek(0)
                        file.write(content)
                        file.truncate()

    print("done")

def assign_aethestic_scores(folder_path):
    processor = AutoImageProcessor.from_pretrained("shadowlilac/aesthetic-shadow-v2")
    model = AutoModelForImageClassification.from_pretrained("shadowlilac/aesthetic-shadow-v2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    aethestic_scores = []

    # load 1024x1024 images
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            with Image.open(file_path) as img:
                inputs = processor(images=img, return_tensors="pt")
                inputs.to(device)
                outputs = model(**inputs) 
                logits = outputs.logits
                probs = logits.softmax(dim=-1)
                aethestic_scores.append({
                    "filename": filename,
                    "score": probs[0][0].item()
                })

    # sort by score
    aethestic_scores.sort(key=lambda x: x["score"], reverse=True)
    print(aethestic_scores)

    # unload model
    del model
    del processor
    return aethestic_scores

def combine_tags_from_meta_to_dataset(meta_folder_path, folder_path):
    # search for caption file (.txt) in meta folder, read the content then merge with the file of the same name in the dataset folder
    for filename in tqdm(os.listdir(meta_folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(meta_folder_path, filename)
            with open(file_path, "r") as file:
                meta_content = file.read()
                for file in os.listdir(folder_path):
                    if filename in file and file.endswith(".txt"):
                        file_path_2 = os.path.join(folder_path, file)
                        with open(file_path_2, "r+") as file_2:
                            content = file_2.read()
                            content = meta_content + ", " + content
                            content = content.replace("_", " ")
                            # split by comma, remove duplicates, join back with comma
                            tags = content.split(", ")
                            tags = list(dict.fromkeys(tags))
                            content = ", ".join(tags)
                            file_2.seek(0)
                            file_2.write(content)
                            file_2.truncate()

# attempt to resize images in folder path to one of the above resolutions by padding with white background so that there is as little padding as possible
def resize_images(folder_path, debug=False):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                # calculate the aspect ratio
                width, height = img.size
                if width == 1024 and height == 1024:
                    continue
                aspect_ratio = width / height
                if debug: print(aspect_ratio)
                # if aspect ratio is nearing 1.0, resize to 1024x1024 and crop, otherwise resize to 1024x1024 with padding
                if aspect_ratio > 0.9 and aspect_ratio < 1.1:
                    if aspect_ratio > 1:
                        new_height = 1024
                        new_width = int(1024 * aspect_ratio) # bigger than 1024
                    else:
                        new_width = 1024
                        new_height = int(1024 / aspect_ratio) # bigger than 1024
                    if debug: print('cropping')
                    new_img = Image.new("RGB", (1024, 1024), (255, 255, 255))
                    new_img.paste(img.resize((new_width, new_height)), ((1024 - new_width) // 2, (1024 - new_height) // 2))
                    new_img.save(file_path)
                else:
                    if aspect_ratio > 1:
                        new_height = int(1024 / aspect_ratio) # smaller than 1024
                        new_width = 1024
                    else:
                        new_width = int(1024 * aspect_ratio) # smaller than 1024
                        new_height = 1024
                    if debug: print('padding')
                    new_img = Image.new("RGB", (1024, 1024), (255, 255, 255))
                    new_img.paste(img.resize((new_width, new_height)), ((1024 - new_width) // 2, (1024 - new_height) // 2))
                    new_img.save(file_path)

def rename_files(folder_path, character_tag):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r+") as file:
                # replace "1girl", or "1boy,"
                match_pattern = re.compile(r"1(girl|boy), ")

                content = file.read()
                content = match_pattern.sub("", content)
                content = character_tag + ", " + content
                content = content.replace("_", " ")
                file.seek(0)
                file.write(content)
                file.truncate()

def rename_files_add(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r+") as file:
                content = file.read()
                content = content.replace("_", " ")
                file.seek(0)
                file.write(content)
                file.truncate()

def frequency_count(folder_path):
    tag_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                tags = content.split(", ")
                for tag in tags:
                    if tag in tag_dict:
                        tag_dict[tag] += 1
                    else:
                        tag_dict[tag] = 1
    return tag_dict

def remove_tags_from_files(folder_path, tags):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r+") as file:
                content = file.read()
                src_tags = content.split(", ")
                for tag in tags:
                    if tag in src_tags:
                        src_tags.remove(tag)
                content = ", ".join(src_tags)
                file.seek(0)
                file.write(content)
                file.truncate()

def replace_tag_if_found_tag(find_tag, tag_to_be_replaced, replace_tag, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r+") as file:
                content = file.read()
                if find_tag + ', ' in content:
                    content = content.replace(tag_to_be_replaced, replace_tag)
                    file.seek(0)
                    file.write(content)
                    file.truncate()

def remove_img_if_contain_tags(tags, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                content = file.read()
                src_tags = content.split(", ")
                for tag in tags:
                    if tag in src_tags:
                        os.remove(file_path)
                        # image file with the same name
                        img_file_path = os.path.join(folder_path, filename.replace(".txt", ".png"))
                        os.remove(img_file_path)
                        break

# remove all .txt file without a .png file of the same name
def remove_txt_without_img(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            img_file_path = os.path.join(folder_path, filename.replace(".txt", ".png"))
            img_file_path_jpg = os.path.join(folder_path, filename.replace(".txt", ".jpg"))
            img_file_path_jpeg = os.path.join(folder_path, filename.replace(".txt", ".jpeg"))
            if not os.path.exists(img_file_path) and not os.path.exists(img_file_path_jpg) and not os.path.exists(img_file_path_jpeg):
                os.remove(file_path)

# write function to remove randomly a set percentage of images from folder path
# if dropoff is set to True, weight the removal towards the beginning of the list
def remove_images(folder_path, percentage, dropoff=False, pruning=False):
    if pruning:
        file_to_prune = []
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                file_path = os.path.join(folder_path, filename)
                with Image.open(file_path) as img:
                    # calculate the aspect ratio
                    width, height = img.size
                    aspect_ratio = width / height
                    # if aspect ratio is too out of the wack (<0.3 or >3), add to prune list
                    if aspect_ratio < 0.3 or aspect_ratio > 3:
                        file_to_prune.append(filename)
            elif filename.endswith(".gif") or filename.endswith(".webp"):
                file_to_prune.append(filename)

        print(f"- Pruning {len(file_to_prune)} files")
        for file in tqdm(file_to_prune):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            
    if percentage > 0:
        files = os.listdir(folder_path)
        # filter out .txt files
        files = [file for file in files if not file.endswith(".txt")]

        num_files = len(files)
        num_files_to_remove = int(num_files * percentage)
        print(f"Removing {num_files_to_remove} files from {num_files} files")
        files_to_remove = []
        print("- Selecting random files to remove")
        for i in tqdm(range(num_files_to_remove)):
            if dropoff:
                index = int(pow(random.random(), 3) * num_files)
            else:
                index = random.randint(0, num_files - 1)

            # must not remove the same file twice
            while files[index] in files_to_remove:
                if dropoff:
                    index = int(pow(random.random(), 3) * num_files)
                else:
                    index = random.randint(0, num_files - 1)
                    
            files_to_remove.append(files[index])

        # print(files_to_remove)
        print("- Removing {num_files_to_remove} files")
        for file in tqdm(files_to_remove):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)

    print('- Cleaning up directory')
    remove_txt_without_img(folder_path)

def process_simple(character, series, skip_download=True, random_remove=0.0, recent_bias=False, need_manual_pruning=False, need_manual_tag_filter=True, download_src='danbooru', pixiv_search_term = '', use_original = False):
    folder_path="{char}_dataset".format(char=character)
    character_tag="1girl, {char}, {sr}".format(char=character.replace("_", " "), sr=series.replace("_", " "))
    meta_folder_path="data/{char}".format(char=character)

    if not skip_download:
        questionary.print("Crawling images", style="bold fg:green")
        remote_crawl([character, 'solo'], meta_folder_path)
        remote_crawl([character], meta_folder_path, src=download_src, pixiv_search_term=pixiv_search_term)

        # if number of image is too low, do not use CCIP
        should_use_ccip = not (len(os.listdir(meta_folder_path)) < 100)

        questionary.print("Initial processing images with waifuc", style="bold fg:green")
        if use_original:
            questionary.print("Force use original image", style="fg:yellow")
            local_process_raw(meta_folder_path, folder_path)
        
        local_process(meta_folder_path, folder_path, useCCIP=should_use_ccip)

        # if the folder_path still containing 0 image, try local_process again without CCIP
        if should_use_ccip and len(os.listdir(folder_path)) == 0:
            questionary.print("Retry processing images without CCIP", style="bold fg:yellow")
            local_process(meta_folder_path, folder_path, useCCIP=False)
    
    if need_manual_pruning:
        questionary.print("Manual pruning needed, please remove any bad image before continue", style="bold fg:yellow")
        # wait for any key to continue
        input("Press Enter to continue...")

    questionary.print("Removing images", style="bold fg:yellow")

    img_count = len(os.listdir(folder_path)) // 2
    img_count_limit = 500
    final_random_remove = (img_count - img_count_limit) / img_count if img_count > img_count_limit else random_remove

    remove_images(folder_path, final_random_remove, dropoff=recent_bias, pruning=(not need_manual_pruning))

    questionary.print("Resizing images", style="bold fg:green")
    resize_images(folder_path)

    # questionary.print("Calculating aethestic score", style="bold fg:green")
    # assign_aethestic_scores(folder_path)

    questionary.print("Appending character tags", style="bold fg:green")
    assign_extra_tags(meta_folder_path, folder_path, character_tag)

    # replace_tag_if_found_tag("  grand blue", "  grand blue", "grand blue", folder_path)

    # rename_files(folder_path, character_tag)
    # combine_tags_from_meta_to_dataset(meta_folder_path, folder_path)

    freq = frequency_count(folder_path)
    # sort by frequency
    sorted_freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}

    # list out all tag with frequency ratio above 0.2 on console and allow user to select which tag to remove
    removal_candidate = []
    for tag, freq in sorted_freq.items():
        if freq > (0.2 * len(os.listdir(folder_path)) / 2):
            removal_candidate.append(questionary.Choice(title="{tag} ({freq:.2f}%)".format(tag=tag, freq=(freq * 100 / (len(os.listdir(folder_path)) / 2))), value=tag))

    filter_tags = []
    if need_manual_tag_filter:
        filter_tags = questionary.checkbox(
            "Select tags to be removed", choices=removal_candidate
        ).ask()
    else:
        # auto filter rule: all tag with frequency ratio above 0.9 but less than 1.0, all tag include "hair", "eyes", "breast"
        # special tag: always remove "solo" tag, "looking at viewer" should be remove if freq > 0.8 
        for tag, freq in sorted_freq.items():
            if freq > (0.9 * len(os.listdir(folder_path)) / 2) and freq < (1.0 * len(os.listdir(folder_path)) / 2):
                filter_tags.append(tag)
            elif "hair" in tag or "eyes" in tag or "breast":
                filter_tags.append(tag)
            elif tag == "solo":
                filter_tags.append(tag)
            elif "looking at viewer" in tag and freq > (0.8 * len(os.listdir(folder_path)) / 2):
                filter_tags.append(tag)
            
    questionary.print("Removing filtered tags", style="bold fg:green")
    remove_tags_from_files(folder_path, filter_tags)

    questionary.print("Finished!", style="bold fg:green")
    return [{
        "character": character,
        "series": series,
        "dataset": os.path.abspath(folder_path),
        "img_count": len(os.listdir(folder_path)) // 2,
        "character_tag": character_tag,
    }]

def process_batch_waifuc(characters, series_arr, skip_downloads=[], download_src='danbooru', pixiv_search_terms = dict(), use_original = False, skip_waifucs=[], aliases=dict()):
    for character in characters:
        print(f"Crawling {character}")
        # series = series_arr[characters.index(character)]
        skip_download = skip_downloads[characters.index(character)]
        skip_waifuc = skip_waifucs[characters.index(character)]
        alias = aliases[character] if character in aliases else None
        pixiv_search_term = pixiv_search_terms[character] if character in pixiv_search_terms else None

        folder_path="{char}_dataset".format(char=character)
        meta_folder_path="data/{char}".format(char=character)

        if not skip_download:
            questionary.print("Crawling images", style="bold fg:green")
            if download_src == 'pixiv' and pixiv_search_term:
                remote_crawl([character], meta_folder_path, src=download_src, pixiv_search_term=pixiv_search_term)
            else:
                if alias:
                    remote_crawl([alias], meta_folder_path)
                remote_crawl([character, 'order:score'], meta_folder_path)
                remote_crawl([character, 'solo'], meta_folder_path)

        if skip_waifuc:
            continue

        # if number of image is too low, do not use CCIP
        should_use_ccip = not (len(os.listdir(meta_folder_path)) < 100)

        questionary.print("Initial processing images with waifuc", style="bold fg:green")
        if use_original:
            questionary.print("Force use original image", style="fg:yellow")
            local_process_raw(meta_folder_path, folder_path)
        
        local_process(meta_folder_path, folder_path, useCCIP=should_use_ccip)

        # if the folder_path still containing 0 image, try local_process again without CCIP
        if should_use_ccip and len(os.listdir(folder_path)) == 0:
            questionary.print("Retry processing images without CCIP", style="bold fg:yellow")
            local_process(meta_folder_path, folder_path, useCCIP=False)

def process_batch_external(characters, series_arr, random_remove=0.0, recent_bias=False, need_manual_pruning=False, need_manual_tag_filter=True, img_count_limit=-1):

    results = []

    for character in characters:
        print(f"Processing {character}")
        series = series_arr[characters.index(character)]

        folder_path="{char}_dataset".format(char=character)
        character_tag="1girl, {char}, {sr}".format(char=character.replace("_", " "), sr=series.replace("_", " "))
        meta_folder_path="data/{char}".format(char=character)

        if need_manual_pruning:
            questionary.print("Manual pruning needed, please remove any bad image before continue", style="bold fg:yellow")
            # wait for any key to continue
            input("Press Enter to continue...")

        img_count = len(os.listdir(folder_path)) // 2
        img_count_limit = 500
        if img_count > img_count_limit:
            # remove percentage of images until img_count_limit is reached
            random_remove = (img_count - img_count_limit) / img_count
        
        questionary.print("Removing images", style="bold fg:yellow")
        remove_images(folder_path, random_remove, dropoff=recent_bias, pruning=(not need_manual_pruning or img_count > 300))

        questionary.print("Resizing images", style="bold fg:green")
        resize_images(folder_path)

        # questionary.print("Calculating aethestic score", style="bold fg:green")
        # assign_aethestic_scores(folder_path)

        questionary.print("Appending character tags", style="bold fg:green")
        assign_extra_tags(meta_folder_path, folder_path, character_tag)

        # replace_tag_if_found_tag("  grand blue", "  grand blue", "grand blue", folder_path)

        # rename_files(folder_path, character_tag)
        # combine_tags_from_meta_to_dataset(meta_folder_path, folder_path)

        freq = frequency_count(folder_path)
        # sort by frequency
        sorted_freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}

        # list out all tag with frequency ratio above 0.2 on console and allow user to select which tag to remove
        removal_candidate = []
        for tag, freq in sorted_freq.items():
            if freq > (0.2 * len(os.listdir(folder_path)) / 2):
                removal_candidate.append(questionary.Choice(title="{tag} ({freq:.2f}%)".format(tag=tag, freq=(freq * 100 / (len(os.listdir(folder_path)) / 2))), value=tag))

        filter_tags = []
        if need_manual_tag_filter:
            filter_tags = questionary.checkbox(
                "Select tags to be removed", choices=removal_candidate
            ).ask()
        else:
            # auto filter rule: all tag with frequency ratio above 0.9 but less than 1.0, all tag include "hair", "eyes", "breast"
            # special tag: always remove "solo" tag, "looking at viewer" should be remove if freq > 0.8 
            for tag, freq in sorted_freq.items():
                if freq > (0.9 * len(os.listdir(folder_path)) / 2) and freq < (1.0 * len(os.listdir(folder_path)) / 2):
                    filter_tags.append(tag)
                elif "hair" in tag or "eyes" in tag or "breast":
                    filter_tags.append(tag)
                elif tag == "solo":
                    filter_tags.append(tag)
                elif "looking at viewer" in tag and freq > (0.8 * len(os.listdir(folder_path)) / 2):
                    filter_tags.append(tag)
                
        questionary.print("Removing filtered tags", style="bold fg:green")
        remove_tags_from_files(folder_path, filter_tags)

        questionary.print("Finished!", style="bold fg:green")

        results.append({
            "character": character,
            "series": series,
            "dataset": os.path.abspath(folder_path),
            "img_count": len(os.listdir(folder_path)) // 2,
            "character_tag": character_tag,
        })

    return results


# process_simple(
#     character="asanagi_umi",
#     series="class_de_2-banme_ni_kawaii_onnanoko_to_tomodachi_ni_natta",
#     skip_download=False,
#     random_remove=0.0,
#     recent_bias=False,
#     need_manual_pruning=True,
#     need_manual_tag_filter=True,
# )

# remove_images("D:\\AIstuff\\lora_dataprep\\richelieu_(azur_lane)_dataset", 0.5, False)