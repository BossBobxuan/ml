import numpy
import matplotlib.pyplot
data = numpy.loadtxt("./datafile/irisNoLabel.txt",delimiter=",")
def zeroMean(data):
    meanVal = numpy.mean(data,axis = 0)
    newData = data - meanVal
    return newData,meanVal
newData,meanVal = zeroMean(data)
covMat = numpy.cov(newData,rowvar = 0)#当rowvar=0时，表示一行是一个样本，否则，一列代表一个样本
eigVals,eigVect = numpy.linalg.eig(covMat)#求矩阵的特征值与对应的特征向量，特征向量矩阵的V[:,i]代表特征值矩阵w[i]特征值所对应的特征向量
u,sigma,vt = numpy.linalg.svd(newData)
print(numpy.dot(newData.T,newData).shape)
print(covMat.shape)
print(vt)
print(eigVals)
print(eigVect)
eigValIndice = numpy.argsort(eigVals)#返回一个将数据从小到大排列后，原数组元素的下标
print(eigValIndice)
n_eigValIndice = eigValIndice[-1:-3:-1]
n_eigVect = eigVect[:,n_eigValIndice]
lowDDataMat = numpy.dot(newData,n_eigVect)
reconMat = numpy.dot(lowDDataMat,n_eigVect.T) + meanVal
print(n_eigVect)
print(lowDDataMat)
print(data)
print(reconMat)
matplotlib.pyplot.plot(lowDDataMat[:,0],lowDDataMat[:,1],'ro')
matplotlib.pyplot.show()