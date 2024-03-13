import argparse
from pathlib import Path
from datasets.stanford_online_products import SOP
from datasets.cars196 import Cars196
from datasets.cub_200_2011 import CUB
from datasets.inshop import Inshop
from datasets.aliproducts import Aliproducts
from datasets.rp2k import RP2K
from datasets.products10k import Products10K
from datasets.met import MET
from datasets.hnm import HNM
from datasets.finefood import FineFood
from datasets.shopee import Shopee


name_to_dataset_dict = {
    'sop' : SOP,
    'cars' : Cars196,
    'cub' : CUB,
    'inshop' : Inshop,
    'aliproducts' : Aliproducts,
    'rp2k' : RP2K,
    'products10k' : Products10K,
    'met' : MET,
    'hnm' : HNM,
    'finefood' : FineFood,
    'shopee' : Shopee,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=[
            'folders',
            'sop', 
            'cars',
            'cub',
            'inshop',
            'aliproducts',
            'rp2k',
            'products10k',
            'met',
            'hnm',
            'finefood',
            'shopee'
        ],
        default='sop', 
        help='dataset type'
    )
    
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='./', 
        help='save path'
    )
    
    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()
    DatasetClass = name_to_dataset_dict[args.dataset]
    dataset = DatasetClass(save_path=Path(args.save_path))
    dataset.download()
    dataset.prepare()


    