#This file is make to load the data from the dataset
#T
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys

Folder_PATH = "./PIC/";
num=1;
Sub_ID = str(num).rjust(3,'0');
Sub_Folder_PATH = Folder_PATH + Sub_ID+"/"
pathlist = os.listdir(Sub_Folder_PATH)
pathlist.sort();
Gesture = pathlist[1].split('-')[1]
print(Gesture)
Test_Path = Sub_Folder_PATH+"/"+pathlist[1]
print(Test_Path)
#for folderpath in
File_Path = os.listdir(Test_Path);
for i in range(len(File_Path)):
    File_Path[i] = Test_Path+"/"+File_Path[i]
print(File_Path[1])
print(type(File_Path),type(np.asarray(File_Path)),len(File_Path))
T_img = Image.open(File_Path[1])
tr = transforms.ToTensor();
t_t = tr(T_img)
re = transforms.ToPILImage()
r_im= re(t_t)
print(r_im.size)


class Test_Data_Set(Dataset):
    def __init__(self, file_path,gesture):
        self.to_tensor = transforms.ToTensor();
        self.Img_path = np.asarray(file_path);
        self.label = gesture;
        self.data_len = len(self.Img_path)
    def __getitem__(self,index):
        img_name=self.Img_path[index];
        img = Image.open(img_name);
        img_tensor = self.to_tensor(img);
        img_label = self.label;
        return (img_tensor, img_label)
    def __len__(self):
        return self.data_len
TestDataset = Test_Data_Set(File_Path,Gesture)
set_load = DataLoader(dataset=TestDataset,batch_size=1000,shuffle=False);
fig = plt.figure('test _ img _ content ');
for ind,(image,label) in enumerate(set_load):
    for ind in range(20):
        plt.subplot(5,4,ind+1);
        tran = transforms.ToPILImage()
        plt.imshow(tran(image[ind,:,:,:]))
        plt.title("gesture = "+ str(label[ind]))
plt.show()
