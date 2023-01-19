import pandas as pd
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import json
import random 
 
def get_random_proxy(proxies_list): 
	return random.choice(proxies_list)

if __name__ == '__main__':
    SITE_URL = 'https://www.buycott.com/upc/'
    CSV_PATH = Path('/dataset/product_recognition/folds.csv')
    SAVE_PATH = Path('/dataset/buycott_data')
    SAVE_PATH.mkdir(exist_ok=True, parents=True)
    proxy = "http://4721a9917be4b10be62807a2550cd0e6ecdead39:js_render=true&antibot=true@proxy.zenrows.com:8001"
    proxies = {"http": proxy, "https": proxy}

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
        URL = f'{SITE_URL}{UPC}'
        html_page = requests.get(URL, proxies=proxies, verify=False)
        soup = BeautifulSoup(html_page.content, 'html.parser')

        div=soup.find('div',{"class":"centered_image header_image"})

        pbar.set_description(f"{UPC}")

        if div is not None:
            image = div.find('img')
            img_url = image.attrs['src'].split('?')[0]
            image_urls[UPC] = img_url 
        else:
            print(f'Missing : {UPC}')
            missing_upcs.append(UPC)

    print(f'Number of UPCs without images: {len(missing_upcs)/len(UPCS)}')
    with open(SAVE_PATH / "images_by_upc.json", "w") as fp:
        json.dump(a,fp) 