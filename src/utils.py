import torch
import numpy as np
import os
import random
import yaml
import logging
import json
import pandas as pd
from collections import OrderedDict
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torchvision
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity


def batch_grid(images):
    image_grid = torchvision.utils.make_grid(images, nrow=4)
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    return inv_normalize(image_grid)


def calculate_time(start_time, start_epoch, epoch, epochs):
    t = time.time() - start_time
    elapsed = DayHourMinute(t)
    t /= (epoch + 1) - start_epoch  # seconds per epoch
    t = (epochs - epoch - 1) * t
    remaining = DayHourMinute(t)
    return elapsed, remaining


class DayHourMinute(object):
    def __init__(self, seconds):
        self.days = int(seconds // 86400)
        self.hours = int((seconds - (self.days * 86400)) // 3600)
        self.minutes = int((seconds - self.days * 86400 - self.hours * 3600) // 60)


def save_ckp(save_path, model, epoch=0, optimizer=None, best_loss=100, emb_model_only=False):
    if emb_model_only:
        checkpoint = {
            'model': model.state_dict()
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
        }
    torch.save(checkpoint, save_path)


def load_ckp(checkpoint_fpath, model, optimizer=None, remove_module=False, emb_model_only=False):
    checkpoint = torch.load(checkpoint_fpath)

    pretrained_dict = checkpoint['model']
    if emb_model_only:
        model.load_state_dict(pretrained_dict)
        return model

    model_state_dict = model.state_dict()
    if remove_module:
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        pretrained_dict = new_state_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
    model_state_dict.update(pretrained_dict)
 
    model.load_state_dict(pretrained_dict)

    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        print('Cannot load optimizer params')
        

    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    best_loss = checkpoint['best_loss'] if 'best_loss' in checkpoint else 100
    
    return model, optimizer, epoch, best_loss


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_mapper(mapper_path):
    with open(mapper_path) as file:
        mapper = json.load(file)
    
    return mapper

def calculate_autoscale(train_n_classes):
    return np.sqrt(2) * np.log(train_n_classes-1) 


def get_train_val_split(split_file, fold=0):
    if isinstance(split_file, list):
        df_train = []
        df_valid = []
        df_folds = []
        for sf in split_file:
            df_folds.append(pd.read_csv(sf))
            df_train.append(df_folds[((df_folds.fold != fold) & (df_folds.fold >= 0)) | (df_folds.fold == -1)])
            df_valid.append(df_folds[((df_folds.fold == fold) & (df_folds.fold >= 0)) | (df_folds.fold == -2)])
    else:
        df_folds = pd.read_csv(split_file)
        df_train = df_folds[((df_folds.fold != fold) & (df_folds.fold >= 0)) | (df_folds.fold == -1)]
        df_valid = df_folds[((df_folds.fold == fold) & (df_folds.fold >= 0)) | (df_folds.fold == -2)]

    return df_train, df_valid, df_folds

def get_cp_save_paths(config):
    best_weights_name = 'debug_best.pt' if config['GENERAL']['DEBUG'] else 'best.pt'
    last_weights_name = 'debug_last.pt' if config['GENERAL']['DEBUG'] else 'last.pt'
    best_embeddings_weights_name = 'debug_best_emb.pt' if config['GENERAL']['DEBUG'] else 'best_emb.pt'
    best_cp_sp = os.path.join(config["MISC"]['WORK_DIR'], best_weights_name)
    last_cp_sp = os.path.join(config["MISC"]['WORK_DIR'], last_weights_name)
    best_emb_cp_sp = os.path.join(config["MISC"]['WORK_DIR'], best_embeddings_weights_name)
    return best_cp_sp, last_cp_sp, best_emb_cp_sp

class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger("Logger")
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)

    def info(self, txt):
        self.logger.info(txt)

    def data_info(self, config, df_full, df_train, df_valid):

        total_n_classes = df_full['label_id'].nunique()
        train_n_classes = df_train['label_id'].nunique()
        valid_n_classes = df_valid['label_id'].nunique()

        total_n_samples = len(df_full)
        train_n_samples = len(df_train)
        valid_n_samples = len(df_valid)

        encoder_type = config['MODEL']['ENCODER_NAME']
        margin_type = config['MODEL']['MARGIN_TYPE']
        embeddings_size = config['MODEL']['EMBEDDINGS_SIZE']
        scale_size = config['MODEL']['SCALE_SIZE']
        margin_m = config['MODEL']['M']

        self.logger.info(f'''
        ============   DATA INFO             ============
        Total N classes           : {total_n_classes}
        Total N classes train     : {train_n_classes}
        Total N classes valid     : {valid_n_classes}
        Total N samples           : {total_n_samples}
        Total N training samples  : {train_n_samples}
        Total N validation samples: {valid_n_samples}

        ============   TRAINING PARAMETERS   ============
        Encoder type              : {encoder_type}
        Margin type               : {margin_type}
        Embeddings size           : {embeddings_size}
        Scale size s              : {scale_size:.2f}
        Margin m                  : {margin_m if not isinstance(margin_m, dict) else 'dynamic'}
        =================================================''')

    def epoch_train_info(self, epoch, train_loss, train_acc, valid_loss, valid_acc, gap_val=None):
        epoch_info_str = f'Epoch: {epoch} Train Loss: {train_loss:.5f} Train Acc: {train_acc:.5f}\n'
        epoch_info_str += f'{" "*37} Valid Loss: {valid_loss:.5f} Valid Acc: {valid_acc:.5f}'
        if gap_val is not None:
            epoch_info_str += f'\n{" "*37} GAP value : {gap_val:.5f}'
        self.logger.info(epoch_info_str)
    
    def epoch_time_info(self, start_time, start_epoch, epoch, num_epochs, workdir_path):
        elapsed, remaining = calculate_time(start_time=start_time, 
                                            start_epoch=start_epoch, 
                                            epoch=epoch, 
                                            epochs=num_epochs)

        self.logger.info(f"Epoch {epoch}/{num_epochs} finishied, saved to {workdir_path} ." + \
                         f"\n{' '*37} Elapsed {elapsed.days:d} days {elapsed.hours:d} hours {elapsed.minutes:d} minutes." + \
                         f"\n{' '*37} Remaining {remaining.days:d} days {remaining.hours:d} hours {remaining.minutes:d} minutes.")

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


def load_config(config_path):
    return yaml.safe_load(open(config_path))


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_everything(seed=28):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_embedings(path_to_embedings):
    data=np.load(path_to_embedings, allow_pickle=True)
    embeddings = data.item().get('embeddings')
    labels = data.item().get('labels')

    return embeddings , labels


def load_embedings_separate(path_to_embedings, path_to_labels=None):
    embeddings = np.load(path_to_embedings)
    if path_to_labels is None:
        return embeddings

    labels = np.load(path_to_labels)
    return embeddings , labels


def plot_embeddings(embeddings, labels, save_dir='./', show=True, n_labels=-1, mapper=None,  method='fast_tsne', n_jobs=4):
    
    labels_set = list(set(labels))
    
    print('Fit data to TSNE')
    if  method=='fast_tsne':
        from MulticoreTSNE import MulticoreTSNE as TSNE
        tsne = TSNE(n_jobs=n_jobs)
        tsne_train = tsne.fit_transform(embeddings)
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_jobs=n_jobs)
        tsne_train = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(16, 16))

    for i, l in tqdm(enumerate(labels_set)):
        if n_labels>0:
            if i>n_labels: break
        xs = tsne_train[np.array(labels) == l, 0]
        ys = tsne_train[np.array(labels) == l, 1]
        point_text = str(l) if mapper is None else mapper[str(l)]
        ax.scatter(xs, ys, label=point_text)
        for x, y in zip(xs, ys):
            plt.annotate(point_text,
                         (x, y),
                         size=8,
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

    ax.legend(bbox_to_anchor=(1.05, 1), fontsize='small', ncol=2)
    if show:
        fig.show()

    save_name = save_dir / 'tsne.png'
    fig.savefig(save_name)


def plot_embeddings_interactive(embeddings, labels, save_dir='./', n_labels=-1, mapper=None, save_name=None, method='fast_tsne', n_jobs=4, n_components=2):
    import plotly.graph_objects as go
    
    labels_set = list(set(labels))
    if n_labels>0:
        labels_set = random.sample(labels_set, n_labels)
        embeddings = embeddings[np.isin(labels, labels_set)]
        labels = labels[np.isin(labels, labels_set)]
    else:
        labels_dict = {}
        for l in labels:
            if l not in labels_dict:
                labels_dict[l] = 1
            else:
                labels_dict[l]+=1
        labels_set = [k for k in labels_dict if labels_dict[k] >= -n_labels]
        print(f'total number of labels to visualize: {len(labels_set)}')
        embeddings = embeddings[np.isin(labels, labels_set)]
        labels = labels[np.isin(labels, labels_set)]

    print(f'Fit data using {method}')
    if method=='fast_tsne':
        from MulticoreTSNE import MulticoreTSNE as TSNE
        tsne = TSNE(n_jobs=n_jobs, n_components=n_components)
        tsne_train = tsne.fit_transform(embeddings)
    elif method=='umap':
        from umap import UMAP
        umap = UMAP(n_jobs=n_jobs, n_components=n_components, init='random', random_state=28)
        tsne_train = umap.fit_transform(embeddings)
    else:
        print('Use sklearn TSNE')
        from sklearn.manifold import TSNE
        tsne = TSNE(n_jobs=n_jobs, n_components=n_components)
        tsne_train = tsne.fit_transform(embeddings)

    fig = go.Figure()
    for i, l in enumerate(labels_set):
        xs = tsne_train[np.array(labels) == l, 0]
        ys = tsne_train[np.array(labels) == l, 1]
        
        color = 'rgba({},{},{},{})'.format(int(255*np.random.rand()),
                                           int(255*np.random.rand()),
                                           int(255*np.random.rand()), 0.8)
        point_text = str(l) if mapper is None else mapper[str(l)]

        if n_components==3:
            zs = tsne_train[np.array(labels) == l, 2]
            fig.add_trace(go.Scatter3d(x=xs,
                                    y=ys,
                                    z=zs,
                                    mode='markers',
                                    marker=dict(color=color,
                                                size=3),
                                    text=point_text,
                                    name=point_text))
        else:
            fig.add_trace(go.Scatter(x=xs,
                                    y=ys,
                                    mode='markers',
                                    marker=dict(color=color,
                                                size=5),
                                    text=point_text,
                                    name=point_text))
    fig.update_layout(
        title=go.layout.Title(text=f"{method} plot",
                              xref="paper",
                              x=0),
        autosize=False,
        width=1000,
        height=1000
    )

    if save_name:
        save_name = save_dir / f'{save_name}.html'
    else:
        save_name = save_dir / f'{method}_{n_components}components.html'
    print('Plot data')
    fig.write_html(save_name)


def cosine_similarity_chunks(X, Y, n_chunks=5, top_n=5):
    ch_sz = X.shape[0]//n_chunks

    best_top_n_vals = None
    best_top_n_idxs = None

    for i in tqdm(range(n_chunks)):
        chunk = X[i*ch_sz:,:] if i==n_chunks-1 else X[i*ch_sz:(i+1)*ch_sz,:]
        cosine_sim_matrix_i = cosine_similarity(chunk, Y)
        best_top_n_vals, best_top_n_idxs = calculate_top_n(cosine_sim_matrix_i,
                                                           best_top_n_vals,
                                                            best_top_n_idxs,
                                                            curr_zero_idx=(i*ch_sz),
                                                            n=top_n)
    return best_top_n_vals, best_top_n_idxs

def calculate_top_n(sim_matrix,best_top_n_vals,
                               best_top_n_idxs,
                               curr_zero_idx=0,
                               n=10):
    n_rows, n_cols = sim_matrix.shape
    total_matrix_vals = sim_matrix
    total_matrix_idxs = np.tile(np.arange(n_rows).reshape(n_rows,1), (1,n_cols)).astype(int) + curr_zero_idx
    if curr_zero_idx>0:
        total_matrix_vals = np.vstack((total_matrix_vals, best_top_n_vals))
        total_matrix_idxs = np.vstack((total_matrix_idxs, best_top_n_idxs))
    res = np.argpartition(total_matrix_vals, -n, axis=0)[-n:]
    res_vals = np.take_along_axis(total_matrix_vals, res, axis=0)
    res_idxs = np.take_along_axis(total_matrix_idxs, res, axis=0)

    del res, total_matrix_idxs, total_matrix_vals
    return res_vals, res_idxs