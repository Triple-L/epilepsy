#!-coding:UTF-8-
from numpy import *
import numpy as np


def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def percentage2n(eigVals,percentage):  #pcaPerc中调用的一个函数
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pcaPerc(dataMat, percentage=1):   # 输入pcaPerc(dataMat数据, percentage=1百分比)
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    n=percentage2n(eigVals,percentage)
    n_eigValIndice=eigValInd[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
    lowData_P=meanRemoved*n_eigVect
    reconMat_P = (lowData_P * n_eigVect.T) + meanVals
    return lowData_P,reconMat_P




def pcaACR(dataMat, n=896):   #指定维数 计算累计贡献率
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    sortArray=np.sort(eigVals)   #升序
    arraySum=sum(sortArray) #总和
    eigValInd = sortArray[:-(n+1):-1]
    eigSum = sum(eigValInd)
    acr = eigSum / arraySum
    return acr


def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowData_N = meanRemoved * redEigVects#transform data into new dimensions
    reconMat_N = (lowData_N * redEigVects.T) + meanVals
    return lowData_N,reconMat_N


