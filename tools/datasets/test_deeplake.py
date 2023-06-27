import deeplake
import argparse
import torch
from torchvision import datasets, transforms, models
from pathlib import Path
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='', help='save path')
    parser.add_argument('--bs',type=int, default=8, help='batch size')
    parser.add_argument('--n_jobs', type=int, default=16, help='number of parallel jobs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_path = Path(args.save_path)

    tform = transforms.Compose([
                                transforms.ToPILImage(), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], 
                                                     [0.5, 0.5, 0.5]),
                              ])
    
    ds = deeplake.load("hub://activeloop/stanford-cars-test")
    # ds = deeplake.load("hub://activeloop/stanford-cars-train")
    n_images = len(ds.images)
    print(f'N images: {n_images}')
    
    dataloader = ds.pytorch(num_workers=args.n_jobs, 
                            batch_size=1, 
                            # transform = {'images': tform},
                            shuffle=False)

    print(ds.tensors.keys())
    
    for data in dataloader:
        image  = data['images'].squeeze(0).numpy()
        label  = data['car_models'].squeeze(0).numpy()
        bboxes = data['boxes'].squeeze(0).numpy()
        print(label)
        print(bboxes)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)

        cv2.imwrite('results/img1.png', image)
        break