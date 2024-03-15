import imagesize
import numpy as np
import pandas as pd
from .base import BaseDataset


class FoldersDataset(BaseDataset):
    def __init__(self, save_path, dataset_folder='./'):
        super(FoldersDataset, self).__init__(
            save_path=save_path,
            dataset_folder=dataset_folder
        )


    def download(self):
        pass


    def prepare(self):
        pass