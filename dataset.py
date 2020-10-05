
import pandas as pd
import numpy as np
import os
import cv2, pydicom
import glob
from torch.utils.data import Dataset
import functools

def traintfms(x=256, y=256):
    tfms = [
        A.OneOf([
            A.RandomBrightnessContrast(p=0.3),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)
        ]),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        A.CLAHE(),
        A.Transpose(),
        A.RandomRotate90(),    
        A.Blur(blur_limit=3)
    ]
    return A.Compose(tfms)

def valtfms(x=256, y=256): 
    tfms = A.Compose([A.Resize(x,y)])
    return tfms

def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

formatted_settings = {
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],}

preprocessing_fn = functools.partial(preprocess_input, **formatted_settings)
    
class CTDataset2D(Dataset):
    def __init__(self,df_path,transforms = A.Compose([A.HorizontalFlip()]),preprocessing=get_preprocessing(preprocessing_fn),size=256,mode='val'):
        df = pd.read_csv(df_path)
        self.df_main = df.values
        if mode=='val':
            self.df = self.df_main
        else:
            self.update_train_df()
            
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.size=size


    def __getitem__(self, idx):
        row = self.df[idx]
        img = pydicom.dcmread(glob.glob(f"data/rsna/rsna-str-pulmonary-embolism-detection/{row[0]}/{row[1]}/*{row[2]}.dcm")[0])
        img = img.pixel_array
        label = row[3:].astype(int)
        label[2:] = label[2:] if label[0]==1 else 0
        if self.transforms:
            img = self.transforms(image=img)['image']
        if self.preprocessing:
            img = self.preprocessing(image=img)['image']
        return img,torch.from_numpy(label.reshape(-1))

    def __len__(self):
        return len(self.df)
    
    def update_train_df(self):
        df0 = self.df_main[self.df_main[:,3]==0]
        df1 = self.df_main[self.df_main[:,3]==1]
        np.random.shuffle(df0)
        self.df = np.concatenate([df0[:len(df1)],df1],axis=0)
