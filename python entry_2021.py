# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 19:03:05 2021

@author: yh
"""

import os
import scipy.io as io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import tensorflow as tf
import math
import keras
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, LSTM, BatchNormalization
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Model
from keras.utils import to_categorical
from keras.models import load_model
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import wfdb
import json
import scipy.signal as signal
import warnings
warnings.filterwarnings('ignore')

def butterBandPassFilter(lowcut, highcut, samplerate, order):
    "生成巴特沃斯带通滤波器"
    semiSampleRate = samplerate*0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b,a = signal.butter(order,[low,high],btype='bandpass')
    return b,a


def danpai(peak_time, heartbeat):
    a1 = []
    c1 = []
    for j in range(len(peak_time)):
        data1 = heartbeat[peak_time[j]-60:peak_time[j]+90]
        if len(data1) != 150:
            c1.append(j)
        elif len(data1) == 150:
            a1.append(data1)
    return a1, c1

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a1 = np.argmax(props, axis=1)
    b1 = np.zeros((len(a1), props.shape[1]))
    b1[np.arange(len(a1)), a1] = 1
    return b1

# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(NpEncoder, self).default(obj)

def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False)
        
"""加载模型"""
model = load_model('D:/fangzhen/ECG/challenge/icbeb/model/base.h5')
model.load_weights('D:/fangzhen/ECG/challenge/icbeb/model/base-weight.h5')

           
def baseline(sample_path):
    record_0 = wfdb.rdrecord(sample_path, sampfrom=0, physical=False, channels=[0, ])
    record_1 = wfdb.rdrecord(sample_path, sampfrom=0, physical=False, channels=[1, ])
    ecg_0 = record_0.d_signal#导联I
    ecg_1 = record_1.d_signal#导联II
    #导联I滤波
    ecg_2 = ecg_0.T
    resampled_ecg = signal.medfilt(ecg_2, [1,41]) #一维中值滤波，核的大小一定要是奇数
    resampled_ecg = signal.medfilt(resampled_ecg, [1,121]) #一维中值滤波，核的大小一定要是奇数
    ecg_3 = ecg_2 - resampled_ecg#看来最后需要将这根线减去
    b,a = butterBandPassFilter(1,50,200,order=5)
    ecg_3 = signal.lfilter(b,a,ecg_3)
    ecg_3 = pd.DataFrame(ecg_3.T)
    ecg_3 = ecg_3.iloc[:,0]
    ecg0 = np.array(ecg_3)
    #导联II滤波
    ecg_4 = ecg_1.T
    resampled_ecg = signal.medfilt(ecg_4, [1,41]) #一维中值滤波，核的大小一定要是奇数
    resampled_ecg = signal.medfilt(resampled_ecg, [1,121]) #一维中值滤波，核的大小一定要是奇数
    ecg_5 = ecg_4 - resampled_ecg#看来最后需要将这根线减去
    b,a = butterBandPassFilter(1,50,200,order=5)
    ecg_5 = signal.lfilter(b,a,ecg_5)
    ecg_5 = pd.DataFrame(ecg_5.T)
    ecg_5 = ecg_5.iloc[:,0]
    ecg1 = np.array(ecg_5)
    #标签
    signal_annotation = wfdb.rdann(sample_path, "atr")
    #直接读取R峰位置
    peak_time = signal_annotation.sample
    #切割为单拍
    test_ecg1, de = danpai(peak_time, ecg0)
    test_ecg2, de = danpai(peak_time, ecg1)   
    test_ecg1 = np.array(test_ecg1)
    test_ecg2 = np.array(test_ecg2)   
    test_ecg3 = np.expand_dims(test_ecg1, axis=2)  
    test_ecg4 = np.expand_dims(test_ecg2, axis=2) 
    """预测"""   
    y_pre = model.predict(test_ecg3)
    y_pre1 = y_pre.tolist()
    y_pre2 = props_to_onehot(y_pre1)
    y_pre3 = [np.argmax(one_hot)for one_hot in y_pre2]
    y_pre4 = np.array(y_pre3)   
    #标签转化为3种结果
    nor = 0
    af = 0
    for i in range(len(y_pre4)):
        if y_pre4[i] == 0 :
            nor = nor + 1
        elif y_pre4[i] == 1 :
            af = af + 1
    end_points = []       
    end_ind = 0
    start_points = []
    for i in range(len(y_pre4)-13): #正常，12的原因是长度要超过12，前5拍后8拍   
        if nor/len(y_pre4) >= 0.8 and y_pre4[i]!=1 and y_pre4[i+1]!=1 and y_pre4[i+2]!=1 and y_pre4[i+3]!=1 and y_pre4[i+4]!=1 and y_pre4[i+5]!=1 and y_pre4[i+6]!=1 and y_pre4[i+7]!=1:#正常
            end_points = []
        elif af/len(y_pre4) >= 0.7 :#持续性房颤
            end_points.append(peak_time[0])
            end_points.append(peak_time[-1])   
            break
        #阵发性，连续5拍是AF确定头
        #阵发性，连续5拍
        elif y_pre4[i] ==1 and y_pre4[i+1] ==1 and y_pre4[i+2] ==1 and y_pre4[i+3] ==1 and y_pre4[i+4] ==1 :
            start_ind = i   
            if i > end_ind :#跳过开头相同的循环#这是头部进来的地方，只要进来就找出头尾固定住
                for j in range(len(y_pre4)-start_ind-13):#去掉前面一段，还要加上8拍长度
                    if y_pre4[start_ind+j+5] ==1 and y_pre4[start_ind+j+6] !=1 and y_pre4[start_ind+j+7] !=1 and y_pre4[start_ind+j+8] !=1 and y_pre4[start_ind+j+9] !=1 and y_pre4[start_ind+j+10] !=1 and y_pre4[start_ind+j+11] !=1 and y_pre4[start_ind+j+12] !=1:
                        end_ind = start_ind+j+5 #加5是因为尾巴至少在5拍以后
                        length = j + 5
                        if length > 15:#房颤持续长度15拍
                            if len(de) == 0:#为了和R峰位置一致
                                start_points.append(start_ind)
                                end_points.append(peak_time[start_ind])
                                end_points.append(peak_time[end_ind])                                                     
                            elif len(de) == 1:
                                if de[0] == 0:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+1])
                                    end_points.append(peak_time[end_ind+1])
                                else:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind])
                                    end_points.append(peak_time[end_ind])
                            elif len(de) ==2:
                                if de[1] == 1:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+2])
                                    end_points.append(peak_time[end_ind+2])
                                else:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+1])
                                    end_points.append(peak_time[end_ind+1])  
                            elif len(de) ==3:
                                if de[1] == 1:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+2])
                                    end_points.append(peak_time[end_ind+2])
                                else:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+1])
                                    end_points.append(peak_time[end_ind+1]) 
                            elif len(de) ==4:
                                start_points.append(start_ind)
                                end_points.append(peak_time[start_ind+2])
                                end_points.append(peak_time[end_ind+2])                                                                                                                          
                            break
                    elif j == len(y_pre4)-start_ind-8:#房颤持续长度15拍
                        if y_pre4[j-1]==1 or y_pre4[j-2]==1:
                            end_ind = len(y_pre4)-1#直接去最后一个索引作为终点
                            if len(de) == 0:#为了和R峰位置一致
                                start_points.append(start_ind)
                                end_points.append(peak_time[start_ind])
                                end_points.append(peak_time[end_ind])
                            elif len(de) == 1:
                                if de[0] == 0:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+1])
                                    end_points.append(peak_time[end_ind+1])
                                else:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind])
                                    end_points.append(peak_time[end_ind+1])
                            elif len(de) ==2:
                                if de[1] == 1:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+2])
                                    end_points.append(peak_time[end_ind+2])
                                else:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+1])
                                    end_points.append(peak_time[end_ind+2])  
                            elif len(de) ==3:
                                if de[1] == 1:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+2])
                                    end_points.append(peak_time[end_ind+3])
                                else:
                                    start_points.append(start_ind)
                                    end_points.append(peak_time[start_ind+1])
                                    end_points.append(peak_time[end_ind+3])  
                            elif len(de) ==4:
                                start_points.append(start_ind)
                                end_points.append(peak_time[start_ind+2])
                                end_points.append(peak_time[end_ind+4])   
                            break
                        else:
                            for k in range(8):
                                if y_pre4[j-8+k] == 1 and y_pre4[j-7+k] != 1:
                                    end_ind = j-8+k
                                    if len(de) == 0:#为了和R峰位置一致
                                        start_points.append(start_ind)
                                        end_points.append(peak_time[start_ind])
                                        end_points.append(peak_time[end_ind])
                                    elif len(de) == 1:
                                        if de[0] == 0:
                                            start_points.append(start_ind)
                                            end_points.append(peak_time[start_ind+1])
                                            end_points.append(peak_time[end_ind+1])
                                        else:
                                            start_points.append(start_ind)
                                            end_points.append(peak_time[start_ind])
                                            end_points.append(peak_time[end_ind])
                                    elif len(de) ==2:
                                        if de[1] == 1:
                                            start_points.append(start_ind)
                                            end_points.append(peak_time[start_ind+2])
                                            end_points.append(peak_time[end_ind+2])
                                        else:
                                            start_points.append(start_ind)
                                            end_points.append(peak_time[start_ind+1])
                                            end_points.append(peak_time[end_ind+1])  
                                    elif len(de) ==3:
                                        if de[1] == 1:
                                            start_points.append(start_ind)
                                            end_points.append(peak_time[start_ind+2])
                                            end_points.append(peak_time[end_ind+2])
                                        else:
                                            start_points.append(start_ind)
                                            end_points.append(peak_time[start_ind+1])
                                            end_points.append(peak_time[end_ind+1])  
                                    elif len(de) ==4:
                                        start_points.append(start_ind)
                                        end_points.append(peak_time[start_ind+2])
                                        end_points.append(peak_time[end_ind+2])   
                                    break
    if end_points != []:
        if start_points != [] and start_points[0] <= 5:
            end_points[0] = peak_time[0]            
    end_points1 =[]
    for i in range(len(end_points)):
        end_points1.append([float(end_points[i]),float(end_points[i])])
    pred_dict = {'predict_endpoints': end_points1}
    return pred_dict






"""预测+保存结果"""
if __name__ == '__main__':
    DATA_PATH = r'D:\fangzhen\ECG\challenge\icbeb\test'
    #测试数据文件夹
    RESULT_PATH = r'D:\fangzhen\ECG\challenge\icbeb\test_result'
    #测试结果文件夹
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)   
        pred_dict = baseline(sample_path)
        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)
        