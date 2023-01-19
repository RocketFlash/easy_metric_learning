import os
from pathlib import Path


if __name__ == '__main__':
    DATASET_PATH = Path('/dataset/buycott_data')
    CLASS_PATHS = list(DATASET_PATH.glob('**/'))[1:]

    # for CLASS_PATH in CLASS_PATHS:
    #     CLASS_NAME = CLASS_PATH.name
    #     CLASS_NAME = CLASS_NAME.replace('site:buycott.com ', '')
    #     CLASS_PATH.rename(DATASET_PATH / CLASS_NAME)
    
    
    for CLASS_PATH in CLASS_PATHS:
        if not any(CLASS_PATH.iterdir()):
            CLASS_PATH.rmdir()

    