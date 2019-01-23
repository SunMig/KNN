from os import listdir

from numpy import *
import  operator;

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]  #shape[0]返回矩阵的行数，shape[1]返回矩阵的列数
    diffMat=tile(inX,(dataSetSize,1))-dataSet #每一行出现矩阵inX一次，一共重复四次有四行
    sqDiffMat=diffMat**2 #对矩阵元素进行平方
    sqDistances=sqDiffMat.sum(axis=1) #对矩阵每一行求和，axis=1
    distances=sqDistances**0.5  #对矩阵元素开平方
    sortedDistIndicies=distances.argsort() #矩阵元素升序排序，这里的升序只是把对应的元素的下标拿了出来
    # print(sortedDistIndicies) #可打印观察下
    classCount={} #定义字典，来存放标签和标签出现的次数
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1 #统计相同标签出现的次数，出现一次就加1
    # print(classCount.items())
    #operator.itemgetter函数，用于获取对象的哪些维的数据，参数为序号
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #标签次数降序排序
    # print(sortedClassCount)
    return sortedClassCount[0][0]


# sorted()函数，对单一列表进行排序是升序排序而且保留原列表的数据
#语法：sorted(iterable,camp,key,reverse)
#iterable是排序的对象，cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
#key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
#reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认)
#例：sort(L,cmp=lambda x,y:camp(x[1],y[1]))    sort(L,key=lambda s:s[2],reverse=True)

# 从文本文件中解析数据
def file2matrix(filename):
    fr=open(filename)
    arraylines=fr.readlines()
    numberLines=len(arraylines)
    returnMat=zeros((numberLines,3))
    classLabelVector=[]
    index=0
    for line in arraylines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1

    return returnMat,classLabelVector
#归一化数据
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSET=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSET=dataSet-tile(minVals,(m,1))
    normDataSET=normDataSET/tile(ranges,(m,1))
    return normDataSET,ranges,minVals


#约会网站测试
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    nornMat,ranges,minVals=autoNorm(datingDataMat)
    m=nornMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0
    for i in range (numTestVecs):
        classifierResult=classify0(nornMat[i,:],nornMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with "+str(classifierResult)+" ,the real answer is "+str(datingLabels[i]))
        if(classifierResult!=datingLabels[i]):errorCount+=1
    print("the total error rate is "+str(errorCount/float(numTestVecs)))

#转化图像为向量的函数
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(linStr[j])
    return returnVect
#手写数字测试函数
def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNamestr=trainingFileList[i]
        fileStr=fileNamestr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])#记住要把数字转换为int类型
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/'+fileNamestr)
    testFileList=listdir('testDigits')
    errorcount=0
    n=len(testFileList)
    for i in range(n):
        fileNamestr=testFileList[i]
        fileStr=fileNamestr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vecoterUnderTest=img2vector('testDigits/'+fileNamestr)
        classifierResult=classify0(vecoterUnderTest,trainingMat,hwLabels,3)
        print("the calssifier came back with: "+str(classifierResult)+" ,the real answer is : "+str(classNumStr))
        if(classifierResult!=classNumStr):
            errorcount+=1
    print("the total error number is : "+str(errorcount))
    print("the total error rate is "+str(float(errorcount/n)))
