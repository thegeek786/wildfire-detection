import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import re
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, path_rgb, path_ir,input_size=254, transform=False):
        self.path_rgb = path_rgb
        self.path_noise = path_ir
        self.angle_array = [90, -90, 180, -180, 270, -270]
        # self.target_size = target_size
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
    
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
        
        self.T = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    def __getitem__(self, index):
        name = os.listdir(self.path_rgb)[index]
        ID = re.findall(r"\d+",name)[0]  
        rgb = Image.open(os.path.join(self.path_rgb, name))
        ir = Image.open(os.path.join(self.path_noise , name))
        ID = int(ID)
        if (1<= ID and ID <=13700):
            y = 0
        elif   (13701	<= ID and ID <=14699) \
            or (15981	<= ID and ID <=19802) \
            or (19900	<= ID and ID <=27183) \
            or (27515	<= ID and ID <=31294) \
            or (31510	<= ID and ID <=33597) \
            or (33930	<= ID and ID <=36550) \
            or (38031	<= ID and ID <=38153) \
            or (41642	<= ID and ID <=45279) \
            or (51207	<= ID and ID <=52286):
                
            y = 1
        else:
            y=2                    
        rgb = self.pil2tensor(rgb)
        ir = self.pil2tensor(ir)
        
        if self.transform is True:            
            rgb = self.T (rgb)
            ir  = self.T (ir)
        return rgb, ir,y
    
    def __len__(self):
        return len(os.listdir(self.path_rgb))
    
class MyDataset_train(Dataset):  # train for cross dataset validation
    def __init__(self, path_rgb, path_ir,input_size=254, transform=False):
        self.path_rgb = path_rgb
        self.path_noise = path_ir
        self.angle_array = [90, -90, 180, -180, 270, -270]
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()    
        self.norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))           
        self.T = transforms.Compose([
        transforms.Resize(input_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    def __getitem__(self, index):
        name = os.listdir(self.path_rgb)[index]
        ID = re.findall(r"\d+",name)[0]  
        rgb = Image.open(os.path.join(self.path_rgb, name))
        ir = Image.open(os.path.join(self.path_noise , name))
        ID = int(ID)
        
        if (1<= ID and ID <=13700):
            y = 0
        else:
            y=1
        
        rgb = self.pil2tensor(rgb)
        ir = self.pil2tensor(ir)
        
        if self.transform is True:
            
            rgb = self.T (rgb)
            ir  = self.T (ir)
        
        return rgb, ir,y
    
    def __len__(self):
        return len(os.listdir(self.path_rgb))
    
def MyDataset_test(path_test,input_size=254, transform=False):        
    pil2tensor = transforms.ToTensor()
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if transform is True:
        test_dataset = datasets.ImageFolder(path_test,T)
    else:
        test_dataset = datasets.ImageFolder(path_test,pil2tensor)
    return test_dataset


