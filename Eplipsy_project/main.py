#!-coding:UTF-8
#！usr/bin/env python
from pylab import *
import readData
import pca
import scipy.io
from matplotlib.pyplot import specgram

if __name__ == '__main__':
    eeg_data = scipy.io.loadmat('Patient_1_preictal_segment_0001.mat')     #Patient_1_preictal_segment_0001.mat
    #print type(eeg_data) #字典结构
    preictal_segment_1 = eeg_data['preictal_segment_1']['data'] #选择数据部分
    data = preictal_segment_1[0][0]
    print data[:][:].shape
    #subplot(3, 1, 3)
    figure(1)
    im = specgram(data[0][:], 20, noverlap=10,Fs=256)
    #pca_data = pca.pca(im,80)
   # print pca_data
    show()

