from PIL import Image
import numpy as np
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import sys
import os
import torch


def build_transform_JBHI(is_train,args):
    if is_train:
        train_transform = A.Compose([
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.PadIfNeeded(min_height=args.input_size, min_width=args.input_size, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
            A.RandomCrop(height=args.input_size, width=args.input_size, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT,value=0,p=0.5),
            A.MedianBlur(blur_limit=7, p=0.3),
            A.GaussNoise(var_limit=(0.38), p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2,0.2), p=0.3),
            A.ToFloat(),
            ToTensorV2(),
            ])
    else:
        train_transform = A.Compose([ A.ToFloat(),ToTensorV2(),])
    return train_transform


class Custom_dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, if_train=True):
        
        self.image_dir = image_dir
        self.transform = transform
        data = pd.read_csv(label_dir)
        self.image_all = data.iloc[:, 0].values
        self.if_train=if_train
        if data.shape[1]==2:
            self.label_all = data.iloc[:, 1].values
            if self.if_train:
                self.label_all = np.where(self.label_all== -1, 0, np.where(self.label_all == 0, -1, self.label_all))  
        else:
            self.label_all = data.iloc[:, 1:].values
            if self.if_train:
                self.label_all  = np.where(self.label_all== -1, 0, np.where(self.label_all == 0, -1, self.label_all)) 
        print(data)

    def __getitem__(self, idx):
        image_name = str(self.image_all[idx])
        label = self.label_all[idx]
        image_dir = os.path.join(self.image_dir, image_name)
        x = Image.open(image_dir)

        if self.transform:
            if isinstance(self.transform,A.Compose):
                x = np.array(x)
                x = self.transform(image=x)
                x = x['image']
            else:
                x=self.transform(x)
        return x,label

    def get_labels(self):
        return self.label_all

    def __len__(self):
        return len(self.label_all)




def build_dataset(set_name,is_train,args,is_valid=False):
    if is_train==True:
        transform = build_transform_JBHI(is_train=is_train, args=args)
        Dataset = Custom_dataset(image_dir='../retinal_dataset/data/train/'+set_name,
                                    label_dir='../retinal_dataset/labels/train/'+set_name+'.csv',
                                    transform=transform,if_train=True)
    else:
        if is_valid:
            transform = build_transform_JBHI(is_train=is_train, args=args)
            Dataset = Custom_dataset(image_dir='../retinal_dataset/data/valid/'+set_name,
                          label_dir='../retinal_dataset/labels/valid/'+set_name+'.csv',
                          transform=transform,if_train=False)
        else:
            transform = build_transform_JBHI(is_train=is_train, args=args)
            Dataset = Custom_dataset(image_dir='../retinal_dataset/data/test/'+set_name,
                          label_dir='../retinal_dataset/labels/test/'+set_name+'.csv',
                          transform=transform,if_train=False)

    return Dataset

#### balance sampling from multiple datasets ######
class RandomSamplerWithReplacement(RandomSampler):
    def __iter__(self):
        while True:
            for idx in torch.randperm(len(self.data_source)).tolist():
                yield idx


class CombinedDataLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in dataloaders]
        self.length=[len(dl) for dl in dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        batch_images = []
        batch_labels = []
        for i, iterator in enumerate(self.iterators):
            try:
                image,label = next(iterator)
            except StopIteration:
                self.iterators[i] = iter(self.dataloaders[i])
                image,label = next(self.iterators[i])
            batch_images.append(image)
            batch_labels.append(label)
        return torch.cat(batch_images),torch.cat(batch_labels)

    def __len__(self):
        return max(self.length)    
#### balance sampling from multiple datasets ######
        

def Balanced_dataloader(args):

    DDR_dataset=build_dataset(set_name='DDR_trainset',is_train=True,args=args)
    ADAM_dataset=build_dataset(set_name='ADAM_trainset',is_train=True,args=args)
    HR_dataset=build_dataset(set_name='HR_trainset',is_train=True,args=args)
    Cataract_dataset=build_dataset(set_name='CAT_trainset',is_train=True,args=args)
    PALM_dataset=build_dataset(set_name='PALM_trainset',is_train=True,args=args)
    REFUGE_dataset=build_dataset(set_name='REFUGE_trainset',is_train=True,args=args)

    loader1 = DataLoader(DDR_dataset, batch_size=3, num_workers=8,
        pin_memory=True,sampler=RandomSamplerWithReplacement(DDR_dataset))
    loader2 = DataLoader(ADAM_dataset, batch_size=3, num_workers=8,
        pin_memory=True,sampler=RandomSamplerWithReplacement(ADAM_dataset))
    loader3 = DataLoader(HR_dataset, batch_size=3, num_workers=8,
        pin_memory=True,sampler=RandomSamplerWithReplacement(HR_dataset)) 
    loader4 = DataLoader(Cataract_dataset, batch_size=2, num_workers=8,
        pin_memory=True,sampler=RandomSamplerWithReplacement(Cataract_dataset))  
    loader5 = DataLoader(PALM_dataset, batch_size=2, num_workers=8,
        pin_memory=True,sampler=RandomSamplerWithReplacement(PALM_dataset))  
    loader6 = DataLoader(REFUGE_dataset, batch_size=3, num_workers=8,
        pin_memory=True,sampler=RandomSamplerWithReplacement(REFUGE_dataset))    

    loaders = [loader1, loader2, loader3, loader4, loader5, loader6]
    combined_iter = CombinedDataLoader(loaders)
    return combined_iter




def test_dataloader(args):
    ADAM_test_dataset=build_dataset(set_name='ADAM_testset',is_train=False,args=args)
    PALM_test_dataset=build_dataset(set_name='PALM_testset',is_train=False,args=args)
    APTOS_test_dataset=build_dataset(set_name='APTOS_dataset',is_train=False,args=args)
    DDR_test_dataset=build_dataset(set_name='DDR_testset',is_train=False,args=args)
    HPMI_test_dataset=build_dataset(set_name='HPMI_dataset',is_train=False,args=args)
    HR_test_dataset=build_dataset(set_name='HR_testset',is_train=False,args=args)
    Cataract_test_dataset=build_dataset(set_name='CAT_testset',is_train=False,args=args)
    ORIGA_test_dataset=build_dataset(set_name='Origa_dataset',is_train=False,args=args)
    REFUGE_test_dataset=build_dataset(set_name='REFUGE_testset',is_train=False,args=args)
    RFMiD_test_dataset=build_dataset(set_name='RFMiD_dataset',is_train=False,args=args)
    ODIR_test_dataset=build_dataset(set_name='ODIR_testset',is_train=False,args=args)

    loader1 = DataLoader(ADAM_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader2 = DataLoader(PALM_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader3 = DataLoader(APTOS_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader4 = DataLoader(DDR_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader5 = DataLoader(HPMI_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader6 = DataLoader(HR_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader7 = DataLoader(Cataract_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader8 = DataLoader(ORIGA_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader9 = DataLoader(REFUGE_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader10 = DataLoader(RFMiD_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader11 = DataLoader(ODIR_test_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    
    dataloaders = {
        'Cataract_testset': {
            'dataloader': loader7,
            'binary_list': [0,1,2],
            'multi_list': []
        },
         'REFUGE_testset': {
            'dataloader': loader9,
            'binary_list': [1],
            'multi_list': []
        },
        'ADAM_testset': {
            'dataloader': loader1,
            'binary_list': [3],
            'multi_list': []
        },
         'HR_testset': {
            'dataloader': loader6,
            'binary_list': [4],
            'multi_list': []
        },
        'PALM_testset': {
            'dataloader': loader2,
            'binary_list': [5],
            'multi_list': []
        },
          'DDR_testset': {
            'dataloader': loader4,
            'binary_list': [],
            'multi_list': [0]
        },
         'RFMiD_testset': {
            'dataloader': loader10,
            'binary_list': [0,3],
            'multi_list': []
        },
         'ORIGA_testset': {
            'dataloader': loader8,
            'binary_list': [1],
            'multi_list': []
        },
        'HPMI_testset': {
            'dataloader': loader5,
            'binary_list': [5],
            'multi_list': []
        },
        'APTOS_testset': {
            'dataloader': loader3,
            'binary_list': [],
            'multi_list': [0]
        },
        'ODIR_testset': {
            'dataloader': loader11,
            'binary_list': [0,1,2,3,4,5],
            'multi_list': [0]
        }
    }

    return dataloaders


def valid_dataloader(args):
    ADAM_valid_dataset=build_dataset(set_name='ADAM_validset',is_train=False,args=args,is_valid=True)
    PALM_valid_dataset=build_dataset(set_name='PALM_validset',is_train=False,args=args,is_valid=True)
    DDR_valid_dataset=build_dataset(set_name='DDR_validset',is_train=False,args=args,is_valid=True)
    HR_valid_dataset=build_dataset(set_name='HR_validset',is_train=False,args=args,is_valid=True)
    Cataract_valid_dataset=build_dataset(set_name='CAT_validset',is_train=False,args=args,is_valid=True)
    REFUGE_valid_dataset=build_dataset(set_name='REFUGE_validset',is_train=False,args=args,is_valid=True)


    loader1 = DataLoader(ADAM_valid_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader2 = DataLoader(PALM_valid_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader3 = DataLoader(DDR_valid_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader4 = DataLoader(HR_valid_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader5 = DataLoader(Cataract_valid_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)
    loader6 = DataLoader(REFUGE_valid_dataset, batch_size=args.batchSize, num_workers=8,
        pin_memory=True,shuffle=False,drop_last=False)


    dataloaders = {
        'Cataract_testset': {
            'dataloader': loader5,
            'binary_list': [0,1,2],
            'multi_list': []
        },
         'REFUGE_testset': {
            'dataloader': loader6,
            'binary_list': [1],
            'multi_list': []
        },
        'ADAM_testset': {
            'dataloader': loader1,
            'binary_list': [3],
            'multi_list': []
        },
         'HR_testset': {
            'dataloader': loader4,
            'binary_list': [4],
            'multi_list': []
        },
        'PALM_testset': {
            'dataloader': loader2,
            'binary_list': [5],
            'multi_list': []
        },
          'DDR_testset': {
            'dataloader': loader3,
            'binary_list': [],
            'multi_list': [0]
        }
    }
    return dataloaders