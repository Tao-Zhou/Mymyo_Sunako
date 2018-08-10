from scipy.io import loadmat
import os
import numpy as np
import torch as th
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import butter, lfilter,filtfilt
def butter_highpass(cutoff,fs,order=3):
    nyq = 0.5*fs
    normal_cutoff  = cutoff / nyq
    b,a=butter(order,normal_cutoff,btype='high')
    return  b,a
def butter_high_pass_filter(data,highcut,fs,order=3):
    b,a = butter_highpass(highcut,fs,order=order)
    y= filtfilt(b,a,data)
    return y
def butter_lowpass(cutoff,fs,order=3):
    nyq = 0.5*fs
    normal_cutoff  = cutoff / nyq
    b,a=butter(order,normal_cutoff,btype='low')
    return b,a
def butter_low_pass_filter(data,lowcut,fs,order=2):
    b,a = butter_lowpass(lowcut,fs,order=order)
    y= filtfilt(b,a,data)
    return y
def butter_bandpass(lowcut,highcut,fs,order=2):
    nyq=0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order,[low,high],btype = 'band')
    return b,a
def butter_band_pass_filter(data,lowcut,highcut,fs,order=2):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)
    y= lfilter(b,a,data)
    return y

path = "./DATA/dba-preprocessed-001"
path_list = os.listdir(path)
file_path_list = []

for filename in path_list:
    file_path_list.append(os.path.join(path,filename))

print(file_path_list)
data = loadmat(file_path_list[4])
val = data['data']
val = np.array(val)
print(np.shape(val))
#show the raw data
zero_m = np.zeros((24,128));
# val_s= np.append(val,zero_m,axis= 0)


# for i in range(8):
    # fig_raw = plt.figure('raw_data %d group' %i );
    # for j in range(16):
        # print(i*16+j+1)
        # plt.subplot(4,4,j+1);
        # plt.plot(np.arange(1000),val[:,i*16+j]);
        # plt.title('the channel %d data'%(i*16+j+1));
        # plt.xlim(0,1024);

# Think about the antialising and the phase response feature of the
# filter, follow the instruction in the
# https://www.researchgate.net/post/Should_I_use_low_pass_filter_for_paraspinal_muscle_EMG_signal_during_a_passive_task
# https://www1.udel.edu/biology/rosewc/kaap686/notes/EMG%20analysis.pdf
# http://cs229.stanford.edu/proj2016spr/report/040.pdf
#the raw data needs rectified using the `IIR filter` bandpath pass in 20Hz to 450 butter_worth

# data_filtered=np.zeros((1000,128));
# for i in range(128):
    # data_filtered[:,i] =  butter_band_pass_filter(val[:,i],20,450,1000,4);
# for i in range(8):
    # fig_raw = plt.figure('Filter_data %d group' %(i+1) );
    # for j in range(16):
        # print(i*16+j+1)
        # plt.subplot(4,4,j+1);
        # plt.plot(np.arange(1000),data_filtered[:,i*16+j]);
        # plt.title('the channel %d data'%(i*16+j+1));
        # plt.xlim(0,1024);

# linear envelope detection

# data_recti = np.where((val<0),-val,val);#rectified
# data_linear=np.zeros((1000,128));
# for i in range(128):
    # data_linear[:,i] =  butter_low_pass_filter(data_recti[:,i],6,1000,4);
# for i in range(8):
    # fig_raw = plt.figure('Filter_data %d group' %(i+1) );
    # for j in range(16):
        # print(i*16+j+1)
        # plt.subplot(4,4,j+1);
        # plt.plot(np.arange(1000),data_linear[:,i*16+j]);
        #plt.plot(np.arange(1000),data_recti[:,i*16+j]);
        # plt.title('the channel %d data'%(i*16+j+1));
        # plt.xlim(0,1024);
# plt.show()

#As we cannot see the similarity in different trals of the experiment data
#using the raw_data picture of a rectifiled picture
pic_data = val;
pic_data = np.where((pic_data>0.0025),0.0025,pic_data)
pic_data = np.where((pic_data<-0.0025),-0.0025,pic_data)
pic_data = np.reshape(pic_data,(1000,8,16));
# ori= val;
# print(np.min(val),np.max(val))
# val = np.where((val<0),-val,val)
#val = np.where((val>-0.0025) & (val<0.0025),val,0)
# data_stream = np.reshape(val,(1000,8,16))
# ori_stream= np.reshape(ori,(1000,8,16))
#####siganl processing
# sensor_1_origin = np.append(val[:,1],[0]*24) # for fast fft speed
# ori_ch_1=  np.append(ori[:,1],[0]*24)
# print(type(ori_ch_1))
# ori_ch_1= butter_high_pass_filter(ori_ch_1,25,1000,4)
# ori_ch_1 = np.where((ori_ch_1<0),-ori_ch_1,ori_ch_1)
# ori_ch_1 = butter_low_pass_filter(ori_ch_1,6,1000,4)
############
# fig_time =  plt.figure('envolope')
# plt.plot(np.arange(1024),ori_ch_1)
####Mean Energy
# Energy_cur = []
# for i in range(1000):
    # summar =  np.sqrt(np.dot(val[i,0:64],val[i,0:64]))
    # Energy_cur.append(summar)
# fig_E = plt.figure('Eng - time Line')


#####
#
# print(len(sensor_1_origin))
# length = 1024
# k = np.arange(length)
# T = length/1000
# freq = k/T # normalization, two-side
# freq_one = freq[range(int(length/2))]
# ifft_sensor_1_origin = np.fft.fft(sensor_1_origin)
# ifft_sensor_1_origin_norm = ifft_sensor_1_origin / length
# ifft_sensor_1_origin_norm = ifft_sensor_1_origin_norm[range(int(length/2))]
# fig_sig= plt.figure('Signal-freq(FFT)')
# plt.plot(freq_one,abs(ifft_sensor_1_origin_norm))
#
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
    frame = pic_data[Num]
    frame = np.multiply(np.divide(np.add(np.divide(frame,0.0025),1),2),255)
    frame_re = cv2.resize(frame,(40,32),interpolation=cv2.INTER_LINEAR)
    img=np.zeros((40,32),dtype=np.uint8)
    img = frame_re.astype(np.uint8)
    cv2.imshow('Img',img)

# frame = data_stream[500]# size : 8*16
#
# print(np.min(frame),np.max(frame))
#insert value ###First Problem
# frame = np.multiply(np.divide(np.add(np.divide(frame,0.0025),1),2),255)
# frame_re = cv2.resize(frame,(8,16),interpolation=cv2.INTER_LINEAR)
# img=np.zeros((8,16),dtype=np.uint8)
# img = frame_re.astype(int)
# print(img)
# cv2.imwrite('Test.jpg', frame_re)
##
# stream = val[:,1]
######
# fig  = plt.figure('sEMG Image')
# plt.imshow(img)
# plt.show()

'''
test1 = val[1,:]
test2 = np.reshape(data_stream[1,:,:],(1,128))
'''
