import os
from google_images_download import google_images_download
import pandas as pd
import numpy as np
from pathlib import Path


def download_images_by_keyword(keyword, limit = 100, output_dir = '.'):
    response = google_images_download.googleimagesdownload()
    response.download({ "keywords": keyword, 
                        "limit": limit, 
                        "output_directory": output_dir,
                        'extract_metadata': True})


if __name__ == '__main__':
    CSV_PATH = Path('/dataset/product_recognition/folds.csv')
    SAVE_PATH = Path('/dataset/buycott_data')
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    MAX_IMAGES = 3
    df = pd.read_csv(CSV_PATH, dtype={'file_name': str,
                                        'label' : str,
                                        'width' : int,
                                        'height': int,
                                        'label_id' : int,
                                        'fold' : int})

    UPCS = np.unique(df['label'])

    for UPC in UPCS:
        try:
            download_images_by_keyword(f'site:buycott.com {UPC}',
                                    limit=MAX_IMAGES,
                                    output_dir=str(SAVE_PATH))
        except:
            print(f'Can not download images for {UPC}')