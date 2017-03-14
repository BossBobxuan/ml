import numpy
import matplotlib.pyplot
data = numpy.loadtxt("./datafile/irisNoLabel.txt",delimiter=",")
print(data.shape)
mu = numpy.array([[6.0,2.7,5.1,1.6],[7.2,3.6,6.1,2.5],[5.2,2.7,3.9,1.4]])
error = mu.copy()
c = numpy.zeros((150,))
temporary = numpy.zeros((3,))
def oDistance(vector1,vector2):
    a = abs(vector1 - vector2)
    return numpy.dot(a,a.T)
while True:
    
    for i in range(150):
        for j in range(3):
            temporary[j] = oDistance(data[i,:],mu[j,:])
        c[i] = numpy.where(temporary == temporary.min())[0][0]
    
    for i in range(3):
        
        mu[i,:] = data[c == i,:].sum(axis=0) / (c == i).sum()#求和时，当设置axis时，会使axis等于的值维度消失，对该维度进行求和
        
    
    error = mu - error
    print(error)
    if abs(error.sum()) < 0.01:
        break
    else:
        error = mu.copy()
    print(mu)
print(mu)
for i in range(3):
    print("属于%d的数据"%i)
    print(data[c == i,:])

