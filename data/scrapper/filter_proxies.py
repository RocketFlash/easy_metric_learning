import pandas as pd
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import json

def get(url, proxy): 
    try: 
        # Send proxy requests to the final URL 
        response = requests.get(url, proxies={'http': f"http://{proxy}"}, timeout=2) 
        print(response.status_code, response.text)
        return True
    except Exception as e: 
        print(e)
        return False 
 
def check_proxies(proxies_list): 
    proxies_info = []
    for proxy in tqdm(proxies_list):
        val = get("http://ident.me/", proxy) 
        proxies_info.append(val)
    proxies = np.array(proxies_list)
    return proxies[np.array(proxies_info)]
 


if __name__ == '__main__':
    PROXIES_FILE = 'proxy_list.txt'

    proxies_list = open(PROXIES_FILE , "r").read().strip().split("\n")
    proxies_filtered = check_proxies(proxies_list)

    with open('proxy_list_filtered.txt', 'a') as the_file:
        for prox in proxies_filtered:
            the_file.write(str(prox)+'\n')