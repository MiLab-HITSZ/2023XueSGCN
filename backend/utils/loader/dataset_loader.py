from backend.pytorch.data import COCO
from backend.pytorch.data import VOC
import torch
import torchvision.transforms as transforms
from backend.gcn.utils import *


def gcn_train_transforms(resize_scale, crop_scale):
    return transforms.Compose([
        
        transforms.Resize((resize_scale, resize_scale)),
        MultiScaleCrop(crop_scale, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def gcn_valid_transforms(resize_scale, crop_scale):
    return transforms.Compose([
        Warp(crop_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_transforms(resize_scale, crop_scale, is_gcn=False):
    if is_gcn:
        return gcn_train_transforms(resize_scale, crop_scale)
    return transforms.Compose([
        
        transforms.Resize(resize_scale),
        
        transforms.RandomResizedCrop(crop_scale),
        
        transforms.RandomHorizontalFlip(),
        
        transforms.ToTensor(),
        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def valid_transforms(resize_scale, crop_scale, is_gcn=False):
    if is_gcn:
        return gcn_valid_transforms(resize_scale, crop_scale)
    return transforms.Compose([
        transforms.Resize(resize_scale),
        
        transforms.CenterCrop(crop_scale),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class DatasetLoader(object):
    
    def __init__(self, category_dir, train_path, valid_path, inp_name=None):
        super(DatasetLoader, self).__init__()
        self.category_dir = category_dir
        self.train_path = train_path
        self.valid_path = valid_path
        self.inp_name = inp_name
        self.is_gcn = inp_name is not None

    
    
    def get_loaders(self, batch_size, resize_scale=512, crop_scale=448, dataset='COCO'):
        if dataset == 'COCO':
            train_dataset = COCO(images_dir=self.train_path,
                                 config_dir=self.category_dir,
                                 transforms=train_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                 inp_name=self.inp_name)
            valid_dataset = COCO(images_dir=self.valid_path,
                                 config_dir=self.category_dir,
                                 transforms=valid_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                 inp_name=self.inp_name)
        else:
            train_dataset = VOC(images_dir=self.train_path,
                                config_dir=self.category_dir,
                                transforms=train_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                inp_name=self.inp_name)
            valid_dataset = VOC(images_dir=self.valid_path,
                                config_dir=self.category_dir,
                                transforms=valid_transforms(resize_scale, crop_scale, is_gcn=self.is_gcn),
                                inp_name=self.inp_name)
        
        batch_size = max(1, min(batch_size, len(train_dataset), len(valid_dataset)))

        shuffle = True
        drop_last = True
        num_workers = 32

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=shuffle
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last, shuffle=shuffle
        )
        return train_loader, valid_loader
