import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transformations(aug_name='soft', image_size=(400, 400)):
    all_transforms = {
        'super_soft': A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'soft': A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=(-15, 15), shift_limit=0.05, p=0.2, border_mode=2),
            A.RandomBrightnessContrast(p=0.1,
                                       brightness_limit=(-0.15, 0.15),
                                       contrast_limit=(-0.15, 0.15)),
            A.Blur(p=0.1, blur_limit=(3, 5)),
            A.Cutout(p=0.1,
                     num_holes=5,
                     max_h_size=10,
                     max_w_size=10),
            A.HueSaturationValue(p=0.1,
                                 hue_shift_limit=(-20, 20),
                                 sat_shift_limit=(-30, 30),
                                 val_shift_limit=(-20, 20)),
            A.GaussNoise(var_limit=5. / 255., p=0.05),
            A.ISONoise(p=0.05,intensity=(0.1, 0.5),
                       color_shift=(0.01, 0.05)),
            A.MotionBlur(p=0.01,
                         blur_limit=(3, 5)),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'test_aug': A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'hflip_tta': A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'scale_up_tta': A.Compose([
            A.Resize(int(image_size[0]*1.3), int(image_size[1]*1.3)),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'scale_down_tta': A.Compose([
            A.Resize(int(image_size[0]/1.3), int(image_size[1]/1.3)),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        'no_aug': A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    }

    image_transforms = all_transforms[aug_name]
    return image_transforms