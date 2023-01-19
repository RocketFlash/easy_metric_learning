import os
import numbers
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import Counter

class MXDataset(Dataset):
    '''
    Modified code from https://github.com/deepinsight/insightface
    '''
    def __init__(self, root_dir, transform, calc_cl_count=False, use_cache=False):
        super(MXDataset, self).__init__()

        import mxnet as mx
        
        self.transform = transform
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        
        prop = open(os.path.join(root_dir, "property"), "r").read().strip().split(',')
        assert len(prop) == 3
        self.num_classes = int(prop[0])

        self.use_cache = use_cache
        self.cached_data = []
        self.labels = []

        if self.use_cache:
            print('Data caching...')
            for im_i in tqdm(self.imgidx):
                s = self.imgrec.read_idx(im_i)
                header, img = mx.recordio.unpack(s)
                label = header.label
                if not isinstance(label, numbers.Number):
                    label = label[0]
                label = torch.tensor(label, dtype=torch.long)
                image = mx.image.imdecode(img).asnumpy()

                if self.transform is not None:
                    sample = self.transform(image=image)
                    image = sample['image']

                self.cached_data.append(image)
                self.labels.append(label)

        if calc_cl_count:
            if not self.use_cache:
                for im_i in tqdm(self.imgidx):
                    s = self.imgrec.read_idx(im_i)
                    header, img = mx.recordio.unpack(s)
                    label = header.label
                    if not isinstance(label, numbers.Number):
                        label = label[0]
                    self.labels.append(int(label))
            self.classes_counts = dict(Counter(self.labels))
        else:
            self.classes_counts = {}

    def __getitem__(self, index):
        if self.use_cache:
            image = self.cached_data[index]
            label = self.labels[index]
        else:
            idx = self.imgidx[index]
            s = self.imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)

            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            label = torch.tensor(label, dtype=torch.long)
            image = mx.image.imdecode(img).asnumpy()

            if self.transform is not None:
                sample = self.transform(image=image)
                image = sample['image']

        return image, label

    def __len__(self):
        return len(self.imgidx)