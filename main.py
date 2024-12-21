from data_prep import process_simple, process_batch_external, process_batch_waifuc
import json
import sys

# regina_mercedes seiken_gakuin_no_maken_tsukai
# tachibana_kaoru_(toosaka_asagi) original
# ciel_(toosaka_asagi) original

if sys.argv[1] == 'batch':
    # read batch.txt file, each line is a character and series, separated by space
    with open('batch.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    characters = []
    series_arr = []
    skip_downloads = []
    skip_waifucs = []
    aliases = dict()
    pixiv_search_terms = dict()

    for line in lines:
        character, series, *extra_option = line.strip().split(' ')
        print(f'Character: {character}')
        print(f'Series: {series}')
        skip_download = True if len(extra_option) > 0 and extra_option[0] == '1' else False
        skip_waifuc = True if len(extra_option) > 1 and extra_option[1] == '1' else False
        alias = extra_option[2].split(';') if len(extra_option) > 2 and extra_option[2] != '' else []

        characters.append(character)
        series_arr.append(series)
        skip_downloads.append(skip_download)
        skip_waifucs.append(skip_waifuc)
        for a in alias:
            if 'pixiv:' in a:
                pixiv_search_terms[character] = a.split(':')[1]
            else:
                aliases[character] = a
    
    # print(f'Characters: {characters}')
    # print(f'Series: {series_arr}')
    # print(f'Skip Downloads: {skip_downloads}')
    # print(f'Skip Waifuc: {skip_waifucs}')

    # for each aliases, if it follow format "pixiv:search_term", then it will be used to search pixiv

    if len(pixiv_search_terms) > 0:
        print(f'Pixiv Search Terms: {pixiv_search_terms}')
        process_batch_waifuc(characters=characters, series_arr=series_arr, skip_downloads=skip_downloads, use_original=False, skip_waifucs=[True] * len(characters), pixiv_search_terms=pixiv_search_terms, download_src='pixiv')

    process_batch_waifuc(characters=characters, series_arr=series_arr, skip_downloads=skip_downloads, use_original=False, skip_waifucs=skip_waifucs, aliases=aliases)
    dataprep_result = process_batch_external(characters=characters, series_arr=series_arr, random_remove=0.0, recent_bias=True, need_manual_pruning=True, need_manual_tag_filter=True)

    # save object as json file
    with open('dataprep_result.json', 'w') as f:
        json.dump(dataprep_result, f, indent=4)
        
else:
    # get command arg
    character = sys.argv[1]
    series = sys.argv[2]

    print(f'Character: {character}')
    print(f'Series: {series}')

    dataprep_result = process_simple(
        character=character,
        series=series,
        skip_download=False,
        random_remove=0.0,
        recent_bias=False,
        need_manual_pruning=True,
        need_manual_tag_filter=True,
        use_original=False,
        download_src='pixiv',
        pixiv_search_term="アリナ・クローバー (Alina Clover)"
    )

    # save object as json file
    with open('dataprep_result.json', 'w') as f:
        json.dump(dataprep_result, f, indent=4)


