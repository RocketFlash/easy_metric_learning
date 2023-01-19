import imagesize
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import os
from paramiko import SSHClient
from scp import SCPClient
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find dublicates')
    parser.add_argument('--dataset_path', default="./", help="path to the dataset")
    parser.add_argument('--dataset_csv', default="./dataset.csv", help="path to dataset.csv file")
    parser.add_argument('--user', default="ubuntu", help="remote server user name")
    parser.add_argument('--ip', default="172.31.26.58", help="remote server ip address")
    parser.add_argument('--remote_path', default="./", help="path to the dataset on remote server")
    parser.add_argument('--ssh_key', default="/home/ubuntu/.ssh/dl.pem", help="path to ssh key")
    args = parser.parse_args()

    DATASET_PATH = Path(args.dataset_path)
    DATASET_FILE = args.dataset_csv
    
    USER = args.ip
    IP = args.ip
    REMOTE_DATASET_PATH = Path(args.remote_path)
    SSH_KEY = args.ssh_key

    df = pd.read_csv(DATASET_FILE, dtype={'label': str,
                                        'file_name': str,
                                        'width': int,
                                        'height': int})
    
    with SSHClient() as ssh:
        ssh.load_system_host_keys()
        ssh.connect(IP, username=USER, key_filename=SSH_KEY)

        with SCPClient(ssh.get_transport()) as scp:
            for index, row in df.iterrows():
                class_folder = DATASET_PATH / row['label']
                class_folder.mkdir(parents=True, exist_ok=True)
                remote_file_path = REMOTE_DATASET_PATH / row['file_name']
                local_file_path = DATASET_PATH / row['file_name']
                
                print(f'Downloading to {file_path}/')
                scp.get(remote_file_path,
                        local_path=local_file_path,
                        recursive=True)


    