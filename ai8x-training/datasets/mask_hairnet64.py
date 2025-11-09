"""
MASK Hairnet Detection Dataset for 64x64 images
"""
import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as album
import cv2

import ai8x


class MaskHairnet64Dataset(Dataset):
    """
    MASK Hairnet detection dataset for 4 different classes.
    Optimized for 64x64 input images.

    Args:
    root_dir (string): Root directory of dataset where MASK Hairnet images are located.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    transform (callable, optional): A function/transform that takes in an image
        and returns a transformed version.
    resize_size(int, int): Width and height of the images to be resized for the dataset.
    augment_data(bool): Flag to augment the data or not. If d_type is `test`, augmentation is
        disabled.
    """

    labels = ['improper_fm_hn', 'proper_fm_hn', 'proper_fm_improper_hn', 'proper_hn_improper_fm']
    label_to_id_map = {k: v for v, k in enumerate(labels)}
    label_to_folder_map = {
        'improper_fm_hn': 'improper_fm_hn',
        'proper_fm_hn': 'proper_fm_hn',
        'proper_fm_improper_hn': 'proper_fm_improper_hn',
        'proper_hn_improper_fm': 'proper_hn_improper_fm'
    }

    def __init__(self, root_dir, d_type, transform=None,
                 resize_size=(64, 64), augment_data=False):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, d_type)

        if not self.__check_mask_hairnet_data_exist():
            self.__print_download_manual()
            sys.exit("Dataset not found!")

        self.__get_image_paths()

        self.album_transform = None
        if d_type == 'train' and augment_data:
            self.album_transform = album.Compose([
                album.GaussNoise(var_limit=(1.0, 20.0), p=0.25),
                album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                album.ColorJitter(p=0.5),
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                album.RandomCrop(height=resize_size[0], width=resize_size[1]),
                album.HorizontalFlip(p=0.5),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])
        if not augment_data or d_type == 'test':
            self.album_transform = album.Compose([
                album.SmallestMaxSize(max_size=int(1.2*min(resize_size))),
                album.CenterCrop(height=resize_size[0], width=resize_size[1]),
                album.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))])

        self.transform = transform

    def __check_mask_hairnet_data_exist(self):
        return os.path.isdir(self.data_dir)

    def __print_download_manual(self):
        print("******************************************")
        print("MASK Hairnet dataset not found! Please make sure your data is in the correct directory structure:")
        print("Make sure that images are in the following directory structure:")
        print("  'data/mask_hairnet/train/improper_fm_hn'")
        print("  'data/mask_hairnet/train/proper_fm_hn'")
        print("  'data/mask_hairnet/train/proper_fm_improper_hn'")
        print("  'data/mask_hairnet/train/proper_hn_improper_fm'")
        print("  'data/mask_hairnet/test/improper_fm_hn'")
        print("  'data/mask_hairnet/test/proper_fm_hn'")
        print("  'data/mask_hairnet/test/proper_fm_improper_hn'")
        print("  'data/mask_hairnet/test/proper_hn_improper_fm'")
        print("******************************************")

    def __get_image_paths(self):
        self.data_list = []

        for label in self.labels:
            image_dir = os.path.join(self.data_dir, self.label_to_folder_map[label])
            if not os.path.exists(image_dir):
                print(f"Warning: Directory {image_dir} does not exist")
                continue

            image_count = 0
            for file_name in sorted(os.listdir(image_dir)):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    file_path = os.path.join(image_dir, file_name)
                    if os.path.isfile(file_path):
                        self.data_list.append((file_path, self.label_to_id_map[label]))
                        image_count += 1

            print(f"Class {label}: {image_count} images")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label = torch.tensor(self.data_list[index][1], dtype=torch.int64)

        image_path = self.data_list[index][0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.album_transform:
            image = self.album_transform(image=image)["image"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_mask_hairnet64_dataset(data, load_train, load_test):
    """
    Load the MASK Hairnet dataset.
    Returns each data sample in 64x64 size.

    Data Augmentation: Train samples are augmented randomly with
        - Additive Gaussian Noise
        - RGB Shift
        - Color Jitter
        - Shift & Scale & Rotate
        - Random Crop
        - Horizontal Flip
    """
    (data_dir, args) = data

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])

    if load_train:
        try:
            train_dataset = MaskHairnet64Dataset(root_dir=data_dir, d_type='train',
                                                 transform=transform, augment_data=True)
            print(f"Successfully loaded training dataset with {len(train_dataset)} samples")
        except Exception as e:
            print(f"Error loading training dataset: {e}")
            train_dataset = None
    else:
        train_dataset = None

    if load_test:
        try:
            test_dataset = MaskHairnet64Dataset(root_dir=data_dir, d_type='test', transform=transform)
            print(f"Successfully loaded test dataset with {len(test_dataset)} samples")
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            test_dataset = None
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'mask_hairnet64',
        'input': (3, 64, 64),
        'output': ('improper_fm_hn', 'proper_fm_hn', 'proper_fm_improper_hn', 'proper_hn_improper_fm'),
        'loader': get_mask_hairnet64_dataset,
    },
]