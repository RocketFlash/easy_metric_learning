import numpy as np
import cv2
import bcolz
import pickle
import mxnet as mx
import argparse
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_bin(path, rootdir, transform, image_size=[112, 112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list


def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-d", "--dataset_path", help="mxnet record file path", default='faces_emore', type=str)
    args = parser.parse_args()
    rec_path = Path(args.dataset_path)
    load_mx_rec(rec_path)

    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']

    for i in range(len(bin_files)):
        load_bin(rec_path/(bin_files[i]+'.bin'),
                 rec_path/bin_files[i],
                 transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))