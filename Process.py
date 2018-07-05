from scipy.io import loadmat
import os
import numpy as np
import torch as th
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Main_Path = "./DATA/dba-preprocessed-"
Pic_Path = "./PIC"
if os.path.exists(Pic_Path):
    print("pic is now existed, can output image data")
else:
    os.mkdir(Pic_Path)
    print ("Ner folder %s established!",Pic_Path)
path_li = []
for ID in range(1,19):
    num = str(ID).rjust(3,'0')
    path_li.append(Main_Path+ num)
    #print (path_li)
for SUBJECT_ID in range(len(path_li)):
    pic_path = "./PIC/" + str(SUBJECT_ID+1).rjust(3,'0')
    if os.path.exists(pic_path):
        print("Folder is now existed, can output image data")
    else:
        os.mkdir(pic_path)
        print ("New folder %s established!",pic_path)
    path_local = path_li[SUBJECT_ID]#true ID - 1
    file_list = os.listdir(path_local)
    file_path  = []
    for filename in file_list:
        file_path.append(os.path.join(path_local,filename))
        file_path.sort()
    for j in range(len(file_path)):
        FILE_PATH = file_path[j]
        FILE_PATH = FILE_PATH[-15:-4]
        #print(FILE_PATH)
        if os.path.exists(pic_path+"/"+FILE_PATH):
            print("Folder is now existed, can output image data")
        else:
            os.mkdir(pic_path+"/"+FILE_PATH)
            print ("New folder %s established!",pic_path)
        o_path = pic_path+"/"+FILE_PATH+"/"
        data = loadmat(file_path[j])
        val_matrix = data['data']
        val_matrix = np.array(val_matrix)
        data_matrix = np.reshape(val_matrix,(1000,8,16))
        for TIME_SERIES in range(1000):
            frame = data_matrix[TIME_SERIES]
            frame = np.multiply(np.add(frame,1),128)
            frame = cv2.resize(frame,(40,32),interpolation=cv2.INTER_LINEAR)
            IMG_NAME = str(TIME_SERIES+1).rjust(4,'0')+".jpg"
            cv2.imwrite(o_path+IMG_NAME,frame)

print("------")
