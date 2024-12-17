from waifuc.action import NoMonochromeAction, FilterSimilarAction, \
    TaggingAction, PaddingAlignAction, PersonSplitAction, FaceCountAction, FirstNSelectAction, \
    CCIPAction, ModeConvertAction, ClassFilterAction, RandomFilenameAction, AlignMinSizeAction, MirrorAction, RatingFilterAction, ThreeStageSplitAction, HeadCountAction, TagFilterAction
from waifuc.export import TextualInversionExporter, SaveExporter
from waifuc.source import DanbooruSource, LocalSource, YandeSource, GelbooruSource, PixivSearchSource
import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

def local_process(input_path, output_path, useCCIP=True):
    source = LocalSource(input_path, newest_first=True)
    actions = [
        ModeConvertAction('RGB', 'white'),
        # pre-filtering for images
        NoMonochromeAction(),  # no monochrome, greyscale or sketch
        ClassFilterAction(['illustration', 'bangumi']),  # no comic or 3d
        
        # RatingFilterAction(['safe', 'r15']),  # filter images with rating, like safe, r15, r18
        FilterSimilarAction('all'),  # filter duplicated images

        # human processing
        # FaceCountAction(1),  # drop images with 0 or >1 faces
        PersonSplitAction(),  # crop for each person
        ThreeStageSplitAction(),
        FaceCountAction(1),
        AlignMinSizeAction(1024),  # align to min size

        # CCIP, filter the character you may not want to see in dataset
        CCIPAction(min_val_count=15) if useCCIP else None,

        # tagging with wd14 v2, if you don't need character tag, set character_threshold=1.01
        TaggingAction(),

        FilterSimilarAction('all'),  # filter again
        # PaddingAlignAction((860, 1896)),  # padding to square
        MirrorAction(),  # mirror image for data augmentation
        # RandomFilenameAction(ext='.png'),  # random rename files
    ]
    source.attach(
        # preprocess images with white background RGB
        *[v for v in actions if v is not None]
    ).export(
        # save to surtr_dataset directory
        TextualInversionExporter(output_path)
    )

def local_process_raw(input_path, output_path):
    source = LocalSource(input_path)
    source.attach(
        # preprocess images with white background RGB
        ModeConvertAction('RGB', 'white'),
        # human processing
        # FaceCountAction(1),  # drop images with 0 or >1 faces
        AlignMinSizeAction(1024),  # align to min size

        # tagging with wd14 v2, if you don't need character tag, set character_threshold=1.01
        # TaggingAction(),

        # RandomFilenameAction(ext='.png'),  # random rename files
    ).export(
        # save to surtr_dataset directory
        TextualInversionExporter(output_path)
    )

# because i dont want to edit the source code of waifuc, i will use this function to filter out pixiv images based on its meta fil
def special_filter():
    def filter_fn(meta):
        if 'source' in meta:
            if 'pixiv' in meta['source']:
                if 'tags' in meta:
                    if 'cosplay' in meta['tags']:
                        return False
        return True

    return filter_fn

def remote_crawl(tags, output_path, src='danbooru', pixiv_search_term=""):
    source = DanbooruSource(tags, min_size=1024, username="neroyuki", api_key=os.environ.get('BOORU_API_KEY'))
    if src == 'yande':
        source = YandeSource(tags, min_size=1024)
    if src == 'gelbooru':
        source = GelbooruSource(tags, min_size=1024)
    if src == 'pixiv':
        source = PixivSearchSource(
            pixiv_search_term,
            refresh_token=os.environ.get('PIXIV_REFRESH_TOKEN'),
        )
    
    source.attach(
        # only 1 head,
        # HeadCountAction(1),
        TagFilterAction([tags[0] + "_(cosplay)"], reversed=True),
    )[:1000].export(
        # save images (with meta information from danbooru site)
        SaveExporter(output_path)
    )

if __name__ == '__main__':
    print('a')
