from array import array
from kNN import *
import operator
from numpy import *

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

groups,labels=createDataSet();
# print("sort result is : "+classify0([0,0],groups,labels,3))
#print(tile(labels,(4,1)))
# sum_group=groups.sum(axis=1)
# sum_group=sum_group.argsort()
# print(sum_group)
# print("this is part2's result...")

datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
#约会网站测试分类器
#datingClassTest()

#手写数字识别的测试，将数字的图像格式转换为分类器使用的向量格式
handwritingClassTest()

