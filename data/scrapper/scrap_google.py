from bs4 import BeautifulSoup
import requests
import re
import urllib
import os
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url,headers=header)),'html.parser')

def main(args):
    CSV_PATH = Path('/dataset/product_recognition/folds.csv')
    SAVE_PATH = Path('/dataset/google_data')
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    MAX_IMAGES = 3
    df = pd.read_csv(CSV_PATH, dtype={'file_name': str,
                                        'label' : str,
                                        'width' : int,
                                        'height': int,
                                        'label_id' : int,
                                        'fold' : int})

    UPCS = np.unique(df['label'])

    for UPC in tqdm(UPCS):
        query = f'upc {UPC}'
        
        save_directory = SAVE_PATH / f'{UPC}'
        save_directory.mkdir(exist_ok=True)
        
        query = query.split()
        query ='+'.join(query)
        url="https://www.google.co.in/search?q="+query+"&tbm=isch"
        header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
        soup = get_soup(url,header)
        print(soup)
        ActualImages=[]# contains the link for Large original images, type of  image
        for a in soup.find_all("img",{"class":"rg_meta"}):
            link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
            ActualImages.append((link,Type))
        print(len(ActualImages))
        for i , (img , Type) in enumerate( ActualImages[0:MAX_IMAGES]):
            try:
                req = urllib.request.Request(img, headers={'User-Agent' : header})
                raw_img = urllib.request.urlopen(req).read()
                print('Here!')
                if len(Type)==0:
                    with open(save_directory / f'image_{i}.jpg', 'wb') as handler:
                        handler.write(raw_img)
                else :
                    with open(save_directory / f'image_{i}.{Type}', 'wb') as handler:
                        handler.write(raw_img)
                f.write(raw_img)
                f.close()
            except Exception as e:
                print("could not load : "+img)
                print(e)

if __name__ == '__main__':
    from sys import argv
    try:
        main(argv)
    except KeyboardInterrupt:
        pass
    sys.exit()