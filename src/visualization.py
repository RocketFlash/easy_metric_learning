import torch
import torchvision
from .transform import get_inverse_transfrom
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from PIL import Image
import ipyplot
import albumentations as A
import ipywidgets as widgets
from IPython.display import display, clear_output


def draw_text(
        img, 
        text,
        font=cv2.FONT_HERSHEY_PLAIN,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 0, 0),
        text_color_bg=(255, 255, 255)
    ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def save_batch_grid(
        images, 
        labels,
        norm_std,
        norm_mean,
        save_dir='./', 
        split='train', 
        save_prefix='',
        batch_index=None
    ):

    save_dir = Path(save_dir)
    inv_norm = get_inverse_transfrom(
        std=norm_std, 
        mean=norm_mean
    )
    images = inv_norm(images)

    all_images = []

    for img, label in zip(images, labels):
        img = (img * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        img = np.ascontiguousarray(img)
        draw_text(
            img, 
            str(label),
            font=cv2.FONT_HERSHEY_PLAIN,
            pos=(0, 0),
            font_scale=1,
            font_thickness=1,
            text_color=(0, 0, 0),
            text_color_bg=(255, 255, 255)
        )
        all_images.append(img)

    all_images = np.array(all_images).astype(np.float32) / 255
    images = torch.from_numpy(all_images.transpose(0, 3, 1, 2))

    image_grid = torchvision.utils.make_grid(
        images, 
        nrow=4,
        pad_value=0, 
        padding=5
    )
    
    save_name = f'{split}_batch'
    if batch_index is not None:
        save_name += f'_{batch_index}'
    
    if save_prefix:
        save_name += f'_{save_prefix}'
        
    save_path = save_dir / f'{save_name}.png'
    torchvision.utils.save_image(image_grid, save_path)


class DatasetViewer:
    def __init__(
        self, 
        dataset_path, 
        dataset_info_csv
    ):
        self.dataset_path = dataset_path
        self.dataset_info_csv = dataset_info_csv

        self.label_groups = self.read_dataset_info()
        self.group_names  = sorted(self.label_groups.groups.keys())
        
        self.image_widget = widgets.Output()
        
        self.next_button = widgets.Button(description='Next label')
        self.next_button.on_click(self.next_label)
        
        self.prev_button = widgets.Button(description='Previous label')
        self.prev_button.on_click(self.prev_label)

        self.next_batch_button = widgets.Button(description='Next batch')
        self.next_batch_button.on_click(self.next_batch)
        
        self.prev_batch_button = widgets.Button(description='Previous batch')
        self.prev_batch_button.on_click(self.prev_batch)

        self.random_batch_button = widgets.Button(description='Random batch')
        self.random_batch_button.on_click(self.random_batch)

        self.show_button = widgets.Button(description='Show')
        self.show_button.on_click(self.update_output)

        self.label_text_area = widgets.Textarea(
            value='62200106548',
            placeholder='Type label name. For multiple labels separate them with comma',
            description='Label:',
        )

        self.engine = widgets.Dropdown(
            options=['opencv', 'pillow'],
            value='opencv',
            description='image reading engine:',
        )
        
        self.max_images = widgets.BoundedIntText(
            value=20,
            min=5,
            max=500,
            step=1,
            description='max number of images:',
        )
        
        self.img_width = widgets.BoundedIntText(
            value=150,
            min=30,
            max=500,
            step=1,
            description='image width:',
        )

        self.batch_size = widgets.BoundedIntText(
            value=10,
            min=1,
            max=100,
            step=1,
            description='Batch size:',
        )
        
        self.use_bbox = widgets.Checkbox(
            value=True,
            description='Use bbox:',
        )
        
        self.use_tabs = widgets.Checkbox(
            value=True,
            description='Use tabs:',
        )

        self.show_hash = widgets.Checkbox(
            value=False,
            description='Show hash:',
        )
        
        self.layout = widgets.VBox([
            self.label_text_area,
            self.engine,
            self.max_images,
            self.img_width,
            self.use_bbox,
            self.use_tabs,
            self.show_hash,
            self.batch_size,
            widgets.HBox([
                self.prev_button, 
                self.next_button, 
            ]),
            widgets.HBox([
                self.prev_batch_button, 
                self.next_batch_button, 
            ]),
            self.random_batch_button,
            self.show_button,
            self.image_widget,
        ])


    
    def read_dataset_info(self):
        dataset_df = pd.read_csv(
            self.dataset_info_csv,
            dtype={
                    'label': str,
                    'file_name': str,
                    'width': int,
                    'height': int,
                    'bbox': str,
                    'hash': str
                }
        )

        return dataset_df.groupby('label')


    def read_images_from_df(
        self,
        df, 
        dataset_path, 
    ):
        images = []
        images_info_strs = []

        resize_transform = A.Compose([
            A.LongestMaxSize(self.img_width.value),
            A.PadIfNeeded(
                min_height=self.img_width.value, 
                min_width=self.img_width.value, 
                border_mode=cv2.BORDER_CONSTANT,
                value=(255,255,255))
        ])
    
        for idx, image_info in df.iterrows():
            image_path = str(dataset_path / image_info.file_name)
    
            if 'bbox' in image_info and self.use_bbox.value:
                bbox = [int(x) for x in image_info.bbox.split(' ')]
            else:
                bbox = None
    
            if self.engine.value=='opencv':
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(image_path)
                image = np.array(image)
                
            if bbox is not None:
                x1, y1, w, h = bbox
                image = image[y1:(y1+h), x1:(x1+w), :]
    
            sample = resize_transform(image=image)
            image = sample['image']
                
            images.append(image)
            
            image_info_str = f'{image_info.file_name}'
            if self.show_hash.value and 'hash' in image_info:
                image_info_str += f'\n hash: {image_info.hash}'

            images_info_strs.append(image_info_str)
                
        return images, images_info_strs


    def show_label_images(self):
        labels = self.parse_labels_from_text_area()
    
        all_images = []
        all_labels = []
        all_images_info_strs = []
        
        for lbl in labels:
            if lbl in self.group_names:
                df_label = self.label_groups.get_group(lbl)
                
                images, images_info_strs = self.read_images_from_df(
                    df_label, 
                    self.dataset_path, 
                )
        
                if images:
                    image_labels = [lbl]*len(images)
                    all_images += images
                    all_labels += image_labels
                    all_images_info_strs += images_info_strs
                    if not self.use_tabs.value:
                        ipyplot.plot_images(
                            images, 
                            image_labels,
                            custom_texts=images_info_strs,
                            max_images=self.max_images.value, 
                            img_width=self.img_width.value
                        )
                else:
                    print(f'There are no images of label {lbl}')
            else:
                print(f'Label {lbl} does not exist in the dataset')
                
        if self.use_tabs.value and all_images:
            ipyplot.plot_class_tabs(
                all_images, 
                all_labels,
                custom_texts=all_images_info_strs,
                max_imgs_per_tab=self.max_images.value, 
                img_width=self.img_width.value
            )

    def parse_labels_from_text_area(self):
        return self.label_text_area.value.replace(" ", "").replace("\n", "").split(',')

    def display(self):
        display(self.layout)

    
    def update_output(self, b):
        with self.image_widget:
            clear_output(wait=True)
            self.show_label_images()
            
    
    def next_label(self, b):
        labels = self.parse_labels_from_text_area()
        if labels[-1] in self.group_names:
            label = labels[-1]
            label_idx = self.group_names.index(label)
            next_idx  = min(label_idx+1, len(self.group_names))
            next_label = self.group_names[next_idx]
            self.label_text_area.value = next_label
        else:
            self.label_text_area.value = self.group_names[0]

    
    def prev_label(self, b):
        labels = self.parse_labels_from_text_area()
        if labels[0] in self.group_names:
            label = labels[0]
            label_idx = self.group_names.index(label)
            prev_idx  = max(label_idx-1, 0)
            prev_label = self.group_names[prev_idx]
            self.label_text_area.value = prev_label
        else:
            self.label_text_area.value = self.group_names[0]


    def next_batch(self, b):
        labels = self.parse_labels_from_text_area()
        bs = self.batch_size.value
        if labels[-1] in self.group_names:
            label = labels[-1]
            label_idx = self.group_names.index(label)
            start_idx = min(label_idx+1, len(self.group_names))
            end_idx   = min(label_idx+1+bs, len(self.group_names))
        else:
            start_idx = 0
            end_idx   = min(bs, len(self.group_names))

        next_labels = self.group_names[start_idx:end_idx]
        self.label_text_area.value = ','.join(next_labels)

    
    def prev_batch(self, b):
        labels = self.parse_labels_from_text_area()
        bs = self.batch_size.value
        if labels[0] in self.group_names:
            label = labels[0]
            label_idx = self.group_names.index(label)
            end_idx = max(label_idx-1, 0)
            start_idx = max(label_idx-1-bs, 0)
        else:
            start_idx = 0
            end_idx = max(bs, 0)

        next_labels = self.group_names[start_idx:end_idx]
        self.label_text_area.value = ','.join(next_labels)


    def random_batch(self, b):
        bs = self.batch_size.value
        
        next_labels = np.random.choice(
            self.group_names, 
            size=bs, 
            replace=False
        )

        self.label_text_area.value = ','.join(next_labels)