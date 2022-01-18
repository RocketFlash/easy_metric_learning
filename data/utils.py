from sklearn.model_selection import StratifiedKFold
import cv2
import numpy as np
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def dhash(image, hashSize=8):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def convert_hash(h):
	return int(np.array(h, dtype="float64"))


def chunk(l, n):
	for i in range(0, len(l), n):
		yield l[i: i + n]


def get_labels_to_ids_map(labels):
    labels_to_ids = {}
    ids_to_labels = {}
    idx = 0

    for label in labels:
        if label not in labels_to_ids:
            labels_to_ids[label] = idx
            ids_to_labels[idx] = label
            idx+=1
    
    return labels_to_ids, ids_to_labels


def get_stratified_kfold(df, k=5, random_state=28):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(skf.split(df['file_name'], df['label_id'])):
        df.loc[test_index, 'fold'] = fold
    df['fold'] = df['fold'].astype(int)
    return df