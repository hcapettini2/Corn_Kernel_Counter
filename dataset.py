### Python
import os
import random
import pandas as pd
import numpy as np
#import copy
#import matplotlib.pyplot as plt
#import seaborn as sns

### Pytorch
import torch
#import torch.nn as nn
from torch.utils.data import Dataset
#from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
#import torch.utils.data as data
#import segmentation_models_pytorch as smp
#import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF


### Images
import cv2
from PIL import Image


# Set seeds for reproducibility (PyTorch, Python, Numpy)
seed = 2013031
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class CustomImageDataset(Dataset):
    """As builsing back the images to the original size is usefull for counting the number of kernels
       I need to pass this information """
    def __init__(self, annotations_file,image_info_file,dataset_path,threshold=1e-4,re_scale=1.,img_transforms=None,mask_transforms=None,I_M_transforms=None):        
        self.kernel_positions = pd.read_csv(annotations_file)
        self.image_info       = pd.read_csv(image_info_file)
        self.dataset_path     = dataset_path
        self.threshold        = threshold
        self.re_scale         = re_scale
        self.img_transforms   = img_transforms
        self.mask_transforms  = mask_transforms
        self.I_M_transforms   = I_M_transforms
    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):      
        img_path = os.path.join(self.dataset_path, self.image_info.iloc[idx, 1])
        image_id = self.image_info.iloc[idx, 0]
        image    = Image.open(img_path) 
        #image    = cv2.imread(img_path)
        #image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image)
        image_shape = (image.size[1],image.size[0])
        #image_shape = image.shape
        count = self.image_info.iloc[idx, 5]
        ### Mask retrieval and processing
        mask_path = img_path[:-4]+"_map.TIFF"
        mask      = cv2.imread(mask_path,flags=cv2.IMREAD_ANYDEPTH)    
        mask = pre_process_mask(mask,count,self.threshold,self.re_scale)
        ### Apply transforms
        # Transforms on images
        if self.img_transforms:
            image = self.img_transforms(image)
        # Transforms on labels
        if self.mask_transforms:
            mask = self.mask_transforms(mask)
        # Transforms on labels and images (exactly the same transform is applied to both)
        if self.I_M_transforms:
            image,mask = self.I_M_transforms([image,mask]) 
        return image,mask,image_shape,count
    
def pre_process_mask(mask,count,threshold=1e-4,re_scale=1.):
    """
    Prepocess the mask, first we threshold it to erase small values
    then we normalize it so it sum to one and finally we can rescale it
    to avoid having close to one values
    """
    ### Threshold the maps
    mask[mask <= threshold] = 0.
    ### Normalize to 1 the sum of the map
    #mask = mask/mask.sum()
    #### Scale the maps
    mask *= re_scale
    return mask


class MyRotations(object):
    """Custom transform which appllies exactly the same transforms
       on images and labels, to that purpose we use functional transforms."""
    def __call__(self, x):
        x_ = x[0]
        y_ = x[1]
        if random.random() > 0.5:
            x_ = TF.hflip(x[0])
            y_ = TF.hflip(x[1])
        
        if random.random() > 0.5:
            x_ = TF.vflip(x[0])
            y_ = TF.vflip(x[1])
        
        if random.random() > 0.25:
            angle = random.randint(0, 180)
            x_ = TF.rotate(x[0],angle)
            y_ = TF.rotate(x[1],angle)
        return x_,y_
    
    
class Random_Cropping(object):
    """ Custom crop for training"""
    def __init__(self, height, width):
        self.height = height
        self.width  = width
        
    def __call__(self,x):
        image = x[0]
        mask =  x[1]
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size =(self.height,self.width))
        image = TF.crop(image,i,j,h,w)
        mask  = TF.crop(mask,i,j,h,w)
        
        return image, mask
    
    
    
class padding(object):
    """Custom transform for padding small images"""
    def __init__(self, height, width):
        self.height = height
        self.width  = width
    def __call__(self,x):
        max_w = self.height 
        max_h = self.width
        image = x[0]
        mask =  x[1]
        imsize = image.shape[1:]

        if imsize[0]<=max_h:
            h_padding = (max_w - imsize[0]) / 2
            l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
            r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        else:
            l_pad = 0
            r_pad = 0
            
        if imsize[1]<=max_w:
            v_padding = (max_h - imsize[1]) / 2
            t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
            b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
        else:
            t_pad = 0
            b_pad = 0
        
        padding = ( int(t_pad),int(l_pad), int(b_pad), int(r_pad))
        
        if (not all(pd == 0 for pd in padding)):          
            image = TF.pad(image, padding)
            mask  = TF.pad(mask, padding)
        return image,mask