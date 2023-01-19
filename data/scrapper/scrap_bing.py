import pandas as pd
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import json
import random  
from bing_image_downloader import downloader


if __name__ == '__main__':
    CSV_PATH = Path('/dataset/product_recognition/folds.csv')
    SAVE_PATH = Path('/dataset/bing_data')
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    LIMIT = 3

    df = pd.read_csv(CSV_PATH, dtype={'file_name': str,
                                      'label' : str,
                                      'width' : int,
                                      'height': int,
                                      'label_id' : int,
                                      'fold' : int})
    
    missing_upcs = []
    UPCS = np.unique(df['label'])

    pbar = tqdm(UPCS)
    image_urls = {}
    for UPC in pbar:
        query_string = f'ebay {UPC}'
        downloader.download(query_string, 
                            limit=LIMIT,  
                            output_dir=SAVE_PATH, 
                            timeout=60, 
                            verbose=True)