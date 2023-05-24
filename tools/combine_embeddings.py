import sys
sys.path.append("./")

import argparse
import os
from pathlib import Path
import numpy as np
from src.utils import cosine_similarity_chunks
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
import faiss
import torch


if __name__ == '__main__':
    file1 = Path('/home/ubuntu/tmp/embeddings/goncalo_test/embeddings_ref.npz')
    file2 = Path('/home/ubuntu/tmp/embeddings/goncalo_test/embeddings_test.npz')

    save_path = Path('/home/ubuntu/tmp/embeddings/goncalo_test/')
    save_path.mkdir(exist_ok=True, parents=True)

    print('Load embeddings')
    
    data1 = np.load(file1)
    embeddings1 = data1['embeddings'].astype(np.float32)
    labels1 = data1['labels']
    file_names1 = data1['file_names']

    data2 = np.load(file2)
    embeddings2 = data2['embeddings'].astype(np.float32)
    labels2 = data2['labels']
    file_names2 = data2['file_names']

    print('Ref shape', embeddings1.dtype)
    print('Test shape', embeddings2.dtype)

    embeddings = np.concatenate((embeddings1, embeddings2), axis=0)
    labels = np.concatenate((labels1, labels2), axis=0)
    file_names = np.concatenate((file_names1, file_names2), axis=0)

    print('Embeddings', embeddings.shape)
    print('Labels', labels.shape)
    print('File_names', file_names.shape)

    np.savez(save_path / 'embeddings_full.npz', 
             embeddings=embeddings, 
             labels=labels,
             file_names=file_names)
 

    
