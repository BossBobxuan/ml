import numpy
import matplotlib.pyplot
data = numpy.loadtxt("./datafile/irisNoLabel.txt",delimiter=",")
print(data.shape)
data = data[:,1:3]
mu = numpy.array([[2.7,5.1],[5,6.1]])
error = mu.copy()
c = numpy.zeros((150,))
temporary = numpy.zeros((2,))
def oDistance(vector1,vector2):
    a = abs(vector1 - vector2)
    return numpy.dot(a,a.T)
while True:
    
    for i in range(150):
        for j in range(2):
            temporary[j] = oDistance(data[i,:],mu[j,:])
        c[i] = numpy.where(temporary == temporary.min())[0][0]
    
    for i in range(2):
        
        mu[i,:] = data[c == i,:].sum(axis=0) / ((c == i).sum() + 0.0001)#求和时，当设置axis时，会使axis等于的值维度消失，对该维度进行求和
        
    
    error = mu - error
    print(error)
    if abs(error.sum()) < 0.01:
        break
    else:
        error = mu.copy()
    print(mu)
print(mu)
for i in range(2):
    print("属于%d的数据"%i)
    print(data[c == i,:])
matplotlib.pyplot.plot(data[c == 0,0],data[c == 0,1],"ro")
matplotlib.pyplot.plot(data[c == 1,0],data[c == 1,1],"bo")
matplotlib.pyplot.plot(mu[0,0],mu[0,1],"b^")
matplotlib.pyplot.plot(mu[1,0],mu[1,1],"r^")
matplotlib.pyplot.show()



