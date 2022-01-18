import pandas as pd
from pathlib import Path
import numpy as np
import cv2
from utils import dhash, convert_hash, chunk, timeit
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import os
import pickle


def process_images(payload):
    hashes = {}

    print("[INFO] starting process {}".format(payload["id"]))

    for imagePath in payload["input_paths"]:
        # load the input image, compute the hash, and conver it
        image = cv2.imread(str(imagePath))
        h = dhash(image)
        h = convert_hash(h)
        # update the hashes dictionary
        l = hashes.get(h, [])
        l.append(imagePath)
        hashes[h] = l

    print("[INFO] process {} serializing hashes".format(payload["id"]))
    f = open(payload["output_path"], "wb")
    f.write(pickle.dumps(hashes))
    f.close()


@timeit
def run_hashes_calculation(image_paths, hashes_path):
    procs =  cpu_count()
    numImagesPerProc = len(image_paths) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))
    chunkedPaths = list(chunk(image_paths, numImagesPerProc))

    payloads = []
    for (i, imagePaths) in enumerate(chunkedPaths):

        outputPath = os.path.sep.join([str(hashes_path), "proc_{}.pickle".format(i)])
        
        data = {
            "id": i,
            "input_paths": imagePaths,
            "output_path": outputPath
        }
        payloads.append(data)

    print(f'total number of chunks: {len(payloads)}')
    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")


@timeit
def combine_hashes(hashes_path):
    hashes = {}
    
    for p in hashes_path.glob('*.pickle'):
        data = pickle.loads(open(p, "rb").read())
        for (tempH, tempPaths) in data.items():
            imagePaths = hashes.get(tempH, [])
            imagePaths.extend(tempPaths)
            hashes[tempH] = imagePaths

    print("[INFO] serializing hashes...")
    f = open(hashes_path / 'all_hashes.pickle', "wb")
    f.write(pickle.dumps(hashes))
    f.close()


if __name__ == '__main__':
    DATASET_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/sirajul_dataset/test_set/')
    CLASSES_FOLDER_PATHS = sorted(list(DATASET_PATH.glob('*/')))
    HASHES_PATH = Path('/home/rauf/datasets/retechlabs/metric_learning/sirajul_dataset/test_set_hashes/')
    HASHES_PATH.mkdir(exist_ok=True)
    
    EXT = '.png'
    CALCULATE_HASHES = True
    COMBINE_HASHES = True
    SAVE_DUBLICATES_VISUALIZATION = False
    DELETE_DUBLICATES = False

    image_paths = []

    if CALCULATE_HASHES:
        for class_folder_path in tqdm(CLASSES_FOLDER_PATHS):
            images = sorted(list(class_folder_path.glob(f'*{EXT}')))
            for img in images:
                image_paths.append(str(img))
        run_hashes_calculation(image_paths, hashes_path=HASHES_PATH)

    if COMBINE_HASHES:
        combine_hashes(HASHES_PATH)

    hashes = pickle.loads(open(HASHES_PATH / 'all_hashes.pickle', "rb").read())
    print(f'len of hashes: {len(hashes)}')

    count_dublicates = 0
    count_total = 0
    for (h, hashedPaths) in tqdm(hashes.items()):
        if len(hashedPaths) > 1:
            montage = None

            for p in hashedPaths:
                count_dublicates+=1
                count_total+=1

                if SAVE_DUBLICATES_VISUALIZATION:
                    image = cv2.imread(p)
                    image = cv2.resize(image, (150, 150))

                    if montage is None:
                        montage = image
                    else:
                        montage = np.hstack([montage, image])

            count_dublicates-=1
            if SAVE_DUBLICATES_VISUALIZATION:
                cv2.imwrite(str(HASHES_PATH / (str(h)+'.png')), montage)
            if DELETE_DUBLICATES:
                for p in hashedPaths[1:]:
                    os.remove(p)
        else:
            count_total+=1

    print(f'number of dublicates       : {count_dublicates}')
    print(f'number of hashed images    : {count_total}')
    print(f'dublicates ratio           : {(count_dublicates / count_total) * 100}')

    # n_imgs = len(list(DATASET_PATH.glob(f'**/*{EXT}')))
    # print(f'actual number of images    : {n_imgs}')