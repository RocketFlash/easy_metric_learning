import torch
import torchvision
from .transform import get_inverse_transfrom
import numpy as np
import cv2
from pathlib import Path


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