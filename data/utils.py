from sklearn.model_selection import StratifiedKFold
import cv2
import numpy as np
import time
import pandas as pd


def get_counts_df(df):
    counts = df['label'].value_counts()
    counts_df = pd.DataFrame({
        'label':counts.index, 
        'frequency':counts.values
    })
    return counts_df


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


def get_stratified_kfold(df, k=5, random_state=28):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(skf.split(df['file_name'], df['label'])):
        df.loc[test_index, 'fold'] = fold
    df['fold'] = df['fold'].astype(int)
    return df


def make_all_training(df):
    train_df = df
    train_df['fold'] = -1

    train_df['fold'] = train_df['fold'].astype(int)
    return train_df


def make_all_testing(df):
    train_df = df
    train_df['fold'] = -2

    train_df['fold'] = train_df['fold'].astype(int)
    return train_df