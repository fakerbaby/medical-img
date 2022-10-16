import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import  train_test_split
# from utils import convert_to_numpy


class ImgData(Dataset):
    def __init__(self, data_dir=r'dataset/',
                 class_num=2,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229,0.224,0.225)
                 ):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment
        
        self.check_files()
        
    
    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        # file_dir = os.path.join(self.data_dir, 'origin')
        label_file = os.path.join(self.data_dir, 'csv/label.csv')
        file_list_path = pd.read_csv(label_file)
        
        fl_train, fl_val= train_test_split(file_list_path, test_size=0.2, random_state=2333)
        self.path_list = fl_train if self.train else fl_val
        self.label_dict = file_list_path
    
    def __len__(self):
        return len(self.path_list)
    
    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out
    
    def __getitem__(self, idx):
        filename = os.path.splitext(self.path_list.iloc[idx, 0])[0]
        path = os.path.join(self.data_dir,'origin', self.path_list.iloc[idx, 0])
        # convert img to numpy firstly
        img = Image.open(path).convert('RGB')
        # img = im.transpose(1,2,0)
        label = self.label_dict.iloc[idx, 1]
        labels = self.to_one_hot(label)
        labels = torch.from_numpy(labels).float()
        
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.Resize((512,512)),
            transforms.RandomRotation(10),
            # transforms.RandomCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)]
        ) if self.train else transforms.Compose([
            # transforms.CenterCrop(128),
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)]
        )
        
        img_tensor = trans(img)
        return img_tensor, labels, filename
        

if __name__ == '__main__':
    a = ImgData(data_dir='../../dataset')
    a.check_files()
    # print(a.path_list, a.label_dict)
    # print(len(a))
    print(a[0])