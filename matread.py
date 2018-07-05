from scipy.io import loadmat
import os
import numpy as np
import torch as th
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
path = "./DATA/dba-preprocessed-001"
path_list = os.listdir(path)
file_path_list = []

for filename in path_list:
    file_path_list.append(os.path.join(path,filename))
data = loadmat(file_path_list[2])
val = data['data']
val = np.array(val)
val = np.where((val>-0.0025) & (val<0.0025),val,0)
print(np.min(val),np.max(val))
data_stream = np.reshape(val,(1000,8,16))
#####siganl processing
sensor_1_origin = np.append(val[:,1],[0]*24) # for fast fft speed
##### Mean Energy
Energy_cur = []
for i in range(1000):
    summar =  np.sqrt(np.dot(val[i,0:64],val[i,0:64]))
    Energy_cur.append(summar)
fig_E = plt.figure('Eng - time Line')
plt.plot(np.arange(1000),val[:,2])

#####

#print(len(sensor_1_origin))
length = 1024
k = np.arange(length)
T = length/1000
freq = k/T # normalization, two-side
freq_one = freq[range(int(length/2))]
ifft_sensor_1_origin = np.fft.fft(sensor_1_origin)
ifft_sensor_1_origin_norm = ifft_sensor_1_origin / length
ifft_sensor_1_origin_norm = ifft_sensor_1_origin_norm[range(int(length/2))]
fig_sig= plt.figure('Signal')
plt.plot(freq_one,abs(ifft_sensor_1_origin_norm))

#####``
def nothing(x):
    pass
cv2.namedWindow('Img')
cv2.createTrackbar('order','Img',0,1000,nothing)
while(1):
    k=cv2.waitKey(1)&0xFF
    if k ==27:
        break
    Num= cv2.getTrackbarPos('order','Img')
    frame = data_stream[Num]
    frame = np.multiply(np.divide(np.add(np.divide(frame,0.0025),1),2),255)
    frame_re = cv2.resize(frame,(40,32),interpolation=cv2.INTER_LINEAR)
    img=np.zeros((40,32),dtype=np.uint8)
    img = frame_re.astype(np.uint8)
    print(img)
    cv2.imshow('Img',img)
frame = data_stream[500]# size : 8*16

print(np.min(frame),np.max(frame))
## insert value ###First Problem
frame = np.multiply(np.divide(np.add(np.divide(frame,0.0025),1),2),255)
frame_re = cv2.resize(frame,(8,16),interpolation=cv2.INTER_LINEAR)
img=np.zeros((8,16),dtype=np.uint8)
img = frame_re.astype(int)
print(img)
cv2.imwrite('Test.jpg', frame_re)
##
stream = val[:,1]
######
fig  = plt.figure('sEMG Image')
plt.imshow(img)
plt.show()
####
'''
test1 = val[1,:]
test2 = np.reshape(data_stream[1,:,:],(1,128))
'''
