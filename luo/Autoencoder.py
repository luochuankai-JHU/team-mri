import xlrd
import scipy.io as scio
import os
import scipy
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import preprocessing
import sklearn.metrics as metrics
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

from keras.utils import multi_gpu_model

####### XDZN_JS

datapath0 = r'/home/chuankailuo/program/miccai/XDZN_JS/data/TRAIN/TRAIN_6689/'
datapath1 = r'/home/chuankailuo/program/miccai/XDZN_JS/data/VAL/VAL_558/'  #  C:/Users/luochuankai/Desktop/trash/data/
datapath2 = r'/home/chuankailuo/program/miccai/XDZN_JS/data/TEST/TEST_8110/'

files0 = os.listdir(datapath0)
files1 = os.listdir(datapath1)
files2 = os.listdir(datapath2)

def pad_list(lst):   #用来补零
    if lst.shape[1] <5000:
        lst0 = np.zeros((1, 5000))
        lst0[:, 0:lst.shape[1]] = lst
        return lst0
    else:
        return lst

def getdata(data):
    data['I'] = pad_list(data['I'])[:, 1000:5000]
    data['II'] = pad_list(data['II'])[:, 1000:5000]
    data['III'] = pad_list(data['III'])[:, 1000:5000]
    data['V1'] = pad_list(data['V1'])[:, 1000:5000]
    data['V2'] = pad_list(data['V2'])[:, 1000:5000]
    data['V3'] = pad_list(data['V3'])[:, 1000:5000]
    data['V4'] = pad_list(data['V4'])[:, 1000:5000]
    data['V5'] = pad_list(data['V5'])[:, 1000:5000]
    data['V6'] = pad_list(data['V6'])[:, 1000:5000]
    data['aVF'] = pad_list(data['aVF'])[:, 1000:5000]
    data['aVL'] = pad_list(data['aVL'])[:, 1000:5000]
    data['aVR'] = pad_list(data['aVR'])[:, 1000:5000]
    return data

fea0 = np.zeros( (1,48001) )  # 12*4000 +1
for file in files0:
    matpath0 = datapath0 + file
    data = scipy.io.loadmat(matpath0)
    file1 = int(re.sub("\D", "", file))
    file2 = [file1]
    file2 = np.array(file2)
    file2 = file2.reshape(1,1)
    data = getdata(data)
    data1 = np.concatenate((data['I'], data['II'], data['III'], data['V1'], data['V2'], data['V3'], data['V4'], data['V5'],data['V6'], data['aVF'], data['aVL'], data['aVR']), axis=1)
    data2 = np.concatenate((file2,data1), axis=1)  #前面加个.mat文件名字的标号
    fea0 = np.concatenate((fea0, data2), axis=0)
fea0 = np.delete(fea0, 0, 0)  # 删除A的第一行 全0行
f0 = fea0[np.lexsort(fea0[:,::-1].T)]   #按照第一列排序
ff0 = np.delete(f0,0, axis = 1)  #删除第一列（因为第一列是A0001。mat里的0001） TRAIN
import pickle
df2=open('/home/chuankailuo/program/miccai/XDZN_JS/code/autoencoder/train48000.npy','wb')
pickle.dump(ff0,df2)
df2.close()


fea1 = np.zeros( (1,48001) )  # 12*4000+1
for file in files1:
    matpath1 = datapath1 + file
    data = scipy.io.loadmat(matpath1)
    file1 = int(re.sub("\D", "", file))
    file2 = [file1]
    file2 = np.array(file2)
    file2 = file2.reshape(1,1)
    data = getdata(data)
    data1 = np.concatenate((data['I'],data['II'],data['III'],data['V1'],data['V2'],data['V3'],data['V4'],data['V5'],data['V6'],data['aVF'],data['aVL'],data['aVR']),axis = 1)
    data2 = np.concatenate((file2,data1), axis=1)
    fea1 = np.concatenate((fea1, data2), axis=0)
fea1 = np.delete(fea1, 0, 0)  # 删除A的第一行 全0行
f1 = fea1[np.lexsort(fea1[:,::-1].T)]
ff1 = np.delete(f1,0, axis = 1)  #val
import pickle
df2=open('/home/chuankailuo/program/miccai/XDZN_JS/code/autoencoder/val48000.npy','wb')
pickle.dump(ff1,df2)
df2.close()


fea2 = np.zeros( (1,48001) )  # 12*491 +1
for file in files2:
    matpath2 = datapath2 + file
    data = scipy.io.loadmat(matpath2)
    file1 = int(re.sub("\D", "", file))
    file2 = [file1]
    file2 = np.array(file2)
    file2 = file2.reshape(1, 1)
    data = getdata(data)
    data1 = np.concatenate((data['I'], data['II'], data['III'], data['V1'], data['V2'], data['V3'], data['V4'],data['V5'], data['V6'], data['aVF'], data['aVL'], data['aVR']), axis=1)
    data2 = np.concatenate((file2, data1), axis=1)
    fea2 = np.concatenate((fea2, data2), axis=0)
fea2 = np.delete(fea2, 0, 0)  # 删除A的第一行 全0行
f2 = fea2[np.lexsort(fea2[:,::-1].T)]
ff2 = np.delete(f2,0, axis = 1)  #test
import pickle
df2=open('/home/chuankailuo/program/miccai/XDZN_JS/code/autoencoder/test48000.npy','wb')
pickle.dump(ff2,df2)
df2.close()


x_train = np.load( "C:/Users/luochuankai/Desktop/trash/val48000.npy" )
x_val = np.load( "/home/chuankailuo/program/miccai/XDZN_JS/code/autoencoder/" )
x_test = np.load( "/home/chuankailuo/program/miccai/XDZN_JS/code/autoencoder/test48000.npy" )

y_train = np.load( "/home/chuankailuo/program/miccai/XDZN_JS/y_train.npy" )
y_test = np.load( "/home/chuankailuo/program/miccai/XDZN_JS/y_test.npy" )   #y_test要重做
y_val = np.load( "/home/chuankailuo/program/miccai/XDZN_JS/y_val.npy" )




# os.environ['CUDA_VISIBLE_DEVICES']='7'
# in order to plot in a 2D figure
encoding_dim = 12

# this is our input placeholder
input_img = Input(shape=(48000,))

# encoder layers
encoded = Dense(24000, activation='relu')(input_img)
encoded = Dense(10000, activation='relu')(encoded)
encoded = Dense(5000, activation='relu')(encoded)
encoded = Dense(2500, activation='relu')(encoded)
encoded = Dense(1250, activation='relu')(encoded)
encoded = Dense(600, activation='relu')(encoded)
encoded = Dense(300, activation='relu')(encoded)
encoded = Dense(100, activation='relu')(encoded)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(25, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(25, activation='relu')(encoder_output)
decoded = Dense(50, activation='relu')(decoded)
decoded = Dense(100, activation='relu')(decoded)
decoded = Dense(300, activation='relu')(decoded)
decoded = Dense(600, activation='relu')(decoded)
decoded = Dense(1250, activation='relu')(decoded)
decoded = Dense(2500, activation='relu')(decoded)
decoded = Dense(5000, activation='relu')(decoded)
decoded = Dense(10000, activation='relu')(decoded)
decoded = Dense(24000, activation='relu')(decoded)
decoded = Dense(48000, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)
# autoencoder = multi_gpu_model(autoencoder, 8)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)
# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
# training
autoencoder.fit(x_train, x_train,
                nb_epoch=300,
                batch_size=100,
                shuffle=True)


encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()