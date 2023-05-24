import argparse
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def main(args):
    weights = args.weights
    image_path = args.image_path

    device = torch.device(args.device)
    model = torch.jit.load(args.weights, map_location=device)
    model.to(device)


    transform = A.Compose([
            A.Resize(170, 170),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sample = transform(image=image)
    image = sample['image']

    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        
        image = image.to(device)
        output = model(image)
        embeddings = output.cpu().numpy()

        print(f'Image path: {image_path}')
        print(f'Embeddings: {embeddings.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='weights path')
    parser.add_argument('--image_path', type=str, default='', help='image path')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    args = parser.parse_args()

    main(args)