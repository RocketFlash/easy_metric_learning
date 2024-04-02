import torch
import numpy as np
import os
import math
import cv2
import random
from os.path import isfile
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from .transform import get_transform
from PIL import Image
import io
import base64
from io import BytesIO as _BytesIO
from jinja2 import Template
from easydict import EasyDict as edict


def is_main_process(accelerator):
    if accelerator is not None:
        if accelerator.is_local_main_process:
            is_main = True
        else:
            is_main = False
    else:
        is_main = True
    return is_main


def get_device(device_str):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device 


def save_ckp(
        save_path, 
        model, 
        epoch=0, 
        optimizer=None, 
        best_criterion_val=None, 
        emb_model_only=False,
        criterion='train.losses.total_loss',
        accelerator=None
    ):
    if accelerator is not None:
        accelerator.wait_for_everyone() 
        model_ = accelerator.unwrap_model(model)

        if emb_model_only and hasattr(model_, "embeddings_net"):
            model_state_dict = model_.embeddings_net.state_dict()
        else:
            model_state_dict = model_.state_dict()
    else:
        if emb_model_only and hasattr(model, "embeddings_net"):
            model_state_dict = model.embeddings_net.state_dict()
        else:
            model_state_dict = model.state_dict()

    checkpoint = {
        'model': model_state_dict
    }

    if not emb_model_only:
        checkpoint.update({
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_criterion_val': best_criterion_val,
            'criterion' : criterion
        })

    if accelerator is not None:
        accelerator.save(checkpoint, save_path)
    else:
        torch.save(checkpoint, save_path)
    

def load_ckp(
        ckp_path, 
        model, 
        optimizer=None, 
        device=None,
        accelerator=None
    ):
    
    if accelerator is not None:
        checkpoint = torch.load(ckp_path, map_location=accelerator.device)
    else:
        checkpoint = torch.load(ckp_path, map_location=device)

    pretrained_dict = checkpoint['model']
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)

    if optimizer is not None:
        try:
            if accelerator is not None:
                optimizer.load_state_dict(checkpoint['optimizer'], map_location=accelerator.device)
            else:
                optimizer.load_state_dict(checkpoint['optimizer'], map_location=device)
        except:
            print('Cannot load optimizer params')
        
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    best_criterion_val = checkpoint['best_criterion_val'] if 'best_criterion_val' in checkpoint else None
    
    return model, optimizer, epoch, best_criterion_val


def load_checkpoint(
        ckp_path, 
        model, 
        optimizer=None, 
        logger=None, 
        mode='resume',
        device=None,
        accelerator=None
    ):

    assert isfile(ckp_path), f"no checkpoint found at : {ckp_path}"

    checkpoint_data = {}
    if mode=='resume':
        if logger is not None: logger.info(f'resume training from: {ckp_path}')
        model, optimizer, epoch, best_criterion_val = load_ckp(
            ckp_path, 
            model, 
            optimizer,
            device=device,
            accelerator=accelerator
        )
        checkpoint_data['model']       = model
        checkpoint_data['optimizer']   = optimizer
        checkpoint_data['start_epoch'] = epoch + 1
        checkpoint_data['best_criterion_val'] = best_criterion_val
        
    elif mode=='emb':
        if logger is not None:  logger.info(f"load embeddings net only from: {ckp_path}")
        
        if hasattr(model, "embeddings_net"):
            model.embeddings_net, _, _, _ = load_ckp(
                ckp_path, 
                model.embeddings_net, 
                device=device, 
                accelerator=accelerator
            )
        else:
            model, _, _, _ = load_ckp(
                ckp_path, 
                model,  
                device=device,
                accelerator=accelerator
            )
        checkpoint_data['model'] = model
        
    elif mode=='weights':
        if logger is not None:  logger.info(f"load weights from: {ckp_path}")
        model, _, _, _ = load_ckp(
            ckp_path, 
            model,
            device=device,
            accelerator=accelerator
        )
        checkpoint_data['model'] = model
    else:
        if logger is not None:  logger.info(f"wrong loading mode")

    return checkpoint_data


def load_model_except_torch(
        weights, 
        model_type='torch', 
        device='cpu',
        logger=None
    ):

    model_info = {
        'model_type': model_type
    }
    
    if model_type=='traced':
        if logger is not None:  logger.info(f"load traced model from: {weights}")
        model = torch.jit.load(weights, map_location=device)
        model = model.to(device)
        model.eval()

    elif model_type=='onnx':
        if logger is not None:  logger.info(f"load onnx model from: {weights}")
        import onnxruntime as ort
        model = ort.InferenceSession(
            weights,
            providers=[
                'CUDAExecutionProvider', 
                'CPUExecutionProvider'
            ]
        )
    elif model_type in ['tf_32', 'tf_16', 'tf_int', 'tf_dyn']:
        if logger is not None:  logger.info(f"load tensorflow model from: {weights}")
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=weights)
        model = interpreter.get_signature_runner()
    elif model_type=='tf_full_int':
        if logger is not None:  logger.info(f"load tensorflow model from: {weights}")
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=weights)
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        model = interpreter.get_signature_runner()
        model_info['input_details'] = input_details
        model_info['output_details'] = output_details
    
    model_info['model'] = model

    return model_info


def is_model_best(
        stats,
        best_criterion_val,
        criterion='train.losses.total_loss',
        criterion_type='loss'
    ):
    criterion_list = criterion.split('.')

    crit_val = stats[criterion_list[0]]
    for crit in criterion_list[1:]:
        crit_val = crit_val[crit]
    current_criterion_val = crit_val

    if criterion_type=='loss':
        is_best = True if best_criterion_val > current_criterion_val else False
    else:
        is_best = True if best_criterion_val < current_criterion_val else False

    return is_best, current_criterion_val


def get_save_paths(work_dir):
    best_weights_name = 'best.pt'
    last_weights_name = 'last.pt'
    best_emb_weights_name = 'best_emb.pt'
    last_emb_weights_name = 'last_emb.pt'

    save_paths = edict(dict(
        best_weights_path=work_dir / best_weights_name,
        last_weights_path=work_dir / last_weights_name,
        best_emb_weights_path=work_dir / best_emb_weights_name,
        last_emb_weights_path=work_dir / last_emb_weights_name
    ))
    
    return save_paths


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


def plot_embeddings(
        embeddings, 
        labels, 
        save_path='./tsne.png', 
        show=True, 
        n_labels=-1, 
        mapper=None,  
        method='fast_tsne', 
        n_jobs=4
    ):
    
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


def plot_embeddings_interactive(
        embeddings, 
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
        random_state=28
    ):
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