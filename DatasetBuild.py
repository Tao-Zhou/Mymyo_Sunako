import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys

class emgdataset(Dataset):
    def __init__(self,file_path):
        self.Path=file_path;
        self.subject_id_list = os.listdir(file_path);
        self.subject_id_list.sort();
        self.img_path=[];
        self.label=[];
        self.to_tensor = transforms.ToTensor();
        for i in range(len(self.subject_id_list)):
            if self.subject_id_list[i]!=".DS_Store":
                self.ID = self.subject_id_list[i];
                self.sur_path = self.Path+self.subject_id_list[i]+"/";
                self.trial_path_li = os.listdir(self.sur_path);
                for j in range(len(self.trial_path_li)):
                    if  self.trial_path_li[j]!=".DS_Store":
                        self.geslabel=self.trial_path_li[j].split('-')[1];
                        self.pic_path = self.sur_path+self.trial_path_li[j]+"/";
                        self.pic_list= os.listdir(self.pic_path);
                        self.pic_list.sort();
                        for k in range(len(self.pic_list)):
                            if  self.pic_list[k]!=".DS_Store":
                                print(self.pic_path+self.pic_list[k]+"completed"+"with label:"+self.geslabel)
                                self.img_path.append(self.pic_path+self.pic_list[k]);
                                self.label.append(self.geslabel)
        self.img_path = np.asarray(self.img_path)
        self.label = np.asarray(self.label)
        self.data_len = len(self.img_path)
    def __getitem__(self,index):
        img_path_open = self.img_path[index];
        img_open = Image.open(img_path_open);
        img_ten = self.to_tensor(img_open);
        img_lab = self.label[index];
        return (img_ten,img_lab)
    def __len__(self):
        return self.data_len;
File_Path = "./PIC/";
TestDataset = emgdataset(File_Path)
set_load = DataLoader(dataset=TestDataset,batch_size=1000,shuffle=False);
