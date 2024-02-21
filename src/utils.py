import torch
import numpy as np
import os
import math
import cv2
import random
import yaml
import json
import pandas as pd
from collections import OrderedDict
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torchvision
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from .transform import get_transform
from PIL import Image
import io
import base64
from io import BytesIO as _BytesIO
from jinja2 import Template


def get_device(device_str):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device 


def get_sample(image_path, 
               img_h=170, 
               img_w=170,
               data_type='general'):
    image = cv2.imread(image_path)
    transform = get_transform('test_aug',
                              data_type=data_type, 
                              image_size=(img_h, img_w))
    augmented = transform(image=image)
    img_torch = augmented['image']
    return img_torch.unsqueeze(0)


def get_images_paths(path):
    pathlib_path = Path(path)
    return [l for l in list(pathlib_path.glob('**/*.jpeg')) + \
                       list(pathlib_path.glob('**/*.jpg')) + \
                       list(pathlib_path.glob('**/*.png'))]


def get_value_if_exist(config, name, default_value=False):
    if name in config:
        return config[name]
    else:
        return default_value


def get_image(image_path):
    image = cv2.imread(str(image_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 


def batch_grid(images):
    image_grid = torchvision.utils.make_grid(images, nrow=4)
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    return inv_normalize(image_grid)



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


def load_ckp(checkpoint_fpath, model, optimizer=None, remove_module=False, emb_model_only=False, device=None):
    checkpoint = torch.load(checkpoint_fpath, map_location=device)

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


def get_cp_save_paths(config, work_dir):
    best_weights_name = 'debug_best.pt' if config.debug else 'best.pt'
    last_weights_name = 'debug_last.pt' if config.debug else 'last.pt'
    best_emb_weights_name = 'debug_best_emb.pt' if config.debug else 'best_emb.pt'
    last_emb_weights_name = 'debug_last_emb.pt' if config.debug else 'last_emb.pt'
    
    best_cp_sp = work_dir / best_weights_name
    last_cp_sp = work_dir / last_weights_name
    best_emb_cp_sp = work_dir / best_emb_weights_name
    last_emb_cp_sp = work_dir / last_emb_weights_name
    
    return best_cp_sp, last_cp_sp, best_emb_cp_sp, last_emb_cp_sp


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


def plot_tiles_similarity(similarity_matrix, save_path='./img.png'):
    similarities = similarity_matrix[0]
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(similarities)
    ax.set_xlabel('tile index')
    ax.set_ylabel('cosine similarity')
    fig.savefig(save_path)
    plt.close(fig)


def plot_embeddings(embeddings, labels, save_path='./tsne.png', show=True, n_labels=-1, mapper=None,  method='fast_tsne', n_jobs=4):
    
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

    fig.savefig(save_path)


def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def plot_embeddings_interactive(embeddings, 
                                labels, 
                                file_names=None,
                                save_dir='./', 
                                n_labels=-1, 
                                mapper=None, 
                                save_name=None,
                                dataset_path='', 
                                method='fast_tsne', 
                                n_jobs=4, 
                                n_components=2, 
                                random_state=28):
    import plotly.graph_objects as go
    
    labels_set = list(set(labels))
    if n_labels>0:
        labels_set = random.sample(labels_set, n_labels)
    else:
        labels_dict = {}
        for l in labels:
            if l not in labels_dict:
                labels_dict[l] = 1
            else:
                labels_dict[l]+=1
        labels_set = [k for k in labels_dict if labels_dict[k] >= -n_labels]
        print(f'total number of labels to visualize: {len(labels_set)}')
    
    labels_mask = np.isin(labels, labels_set)
    embeddings = embeddings[labels_mask]
    labels = labels[labels_mask]
    if file_names is not None:
        file_names = file_names[labels_mask]

        label_to_fname = {}
        for lbl, fnm in zip(labels, file_names):
            if lbl not in label_to_fname:
                label_to_fname[lbl] = fnm

        label_to_image = {}
        if dataset_path:
            dataset_path = Path(dataset_path)
            for k, v in label_to_fname.items():
                img_path = str(dataset_path/v)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (100, 100))
                img = Image.fromarray(img)
                buff = _BytesIO()
                img.save(buff, format='png')
                encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
                label_to_image[k] = encoded

        # Define the HTML legend container outside the plot
        legend_html = '<div class="legend-container">'

        # Loop through each class and create legend items with preview images
        for label, img_data_base64 in label_to_image.items():
            # Convert image data to base64
            img_src = f'data:image/png;base64,{img_data_base64}'
            
            # Add legend item with the label and image
            legend_html += f"""
            <div class="legend-item">
                <img src="{img_src}" width="100" height="100">
                <span>{label}</span>
            </div>
            """

        legend_html += '</div>'

    print(f'Fit data using {method}')
    if method=='fast_tsne':
        from MulticoreTSNE import MulticoreTSNE as TSNE
        tsne = TSNE(n_jobs=n_jobs, n_components=n_components)
        tsne_train = tsne.fit_transform(embeddings)
    elif method=='open_tsne':
        print('Use openTSNE')
        from openTSNE import TSNE
        tsne_train = TSNE(random_state=random_state, n_jobs=n_jobs, n_components=n_components).fit(embeddings)
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
    
    # def update_point(trace, points, selector):
    #     c = list(scatter.marker.color)
    #     s = list(scatter.marker.size)
    #     for i in points.point_inds:
    #         c[i] = '#bae2be'
    #         s[i] = 20
    #         with f.batch_update():
    #             scatter.marker.color = c
    #             scatter.marker.size = s

    fig.update_layout(
        title=go.layout.Title(text=f"{method} plot",
                              xref="paper",
                              x=0),
        autosize=False,
        width=1000,
        height=1000,
        showlegend=False
    )

    fig.add_annotation(
                    x=0,
                    y=1,
                    showarrow=False,
                    text=legend_html,
                    xref="paper",
                    yref="paper",
                    align="left",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    opacity=0.8,
                )


    if save_name:
        save_name = save_dir / f'{save_name}.html'
    else:
        save_name = save_dir / f'{method}_{n_components}components.html'
    # print('Plot data')
    # fig.write_html(save_name)
    # Save the plot as an HTML file with embedded images
    template = Template('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plot with Images</title>
    </head>
    <body>
        <div>
            {{ plot_div }}
        </div>
    </body>
    </html>
    ''')

    # Generate the plotly div
    plot_div = fig.to_html(full_html=False)

    # Combine the plot div and the template
    html_content = template.render(plot_div=plot_div)

    # Save the HTML content to a file
    with open(save_name, 'w') as file:
        file.write(html_content)


def cosine_similarity_chunks(X, Y, n_chunks=5, top_n=5, sparse=False):
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


def show_images(images, n_col=3, save_name=None):
    n_rows = math.ceil(len(images)/n_col)
    fig, ax = plt.subplots(n_rows, n_col, figsize=(25, 12*n_rows))

    for ax_i in ax:
        if len(images) <= n_col:
            ax_i.set_axis_off()
        else:
            for ax_j in ax_i:
                ax_j.set_axis_off()

    if isinstance(images, dict):
        for img_idx, (title, img) in enumerate(images.items()):
            if len(images) <= n_col:
                ax[img_idx].imshow(img)
                ax[img_idx].set_title(title)
            else:
                ax[img_idx//n_col, img_idx%n_col].imshow(img)
                ax[img_idx//n_col, img_idx%n_col].set_title(title)
    else:
        for img_idx, img in enumerate(images):
            if len(images) <= n_col:
                ax[img_idx].imshow(img)
            else:
                ax[img_idx//n_col, img_idx%n_col].imshow(img)

    fig.subplots_adjust(wspace=0, hspace=0)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)
        plt.close(fig)