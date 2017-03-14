import numpy
import matplotlib.pyplot
data = numpy.loadtxt("testSet.txt")
x = numpy.ones((data.shape[0],3))
x[:,1:3] = data[:,0:2]
y = data[:,2]
def qtx(x,q):
    return numpy.dot(q,x.T)
def sigmoidfunc(mx):
    return 1 / (1 + numpy.exp(-mx))
def h(x):
    return sigmoidfunc(qtx(x,q))
q = numpy.zeros((1,3))
alpha = 0.001
epsilon = 0.0001
maxloop = 40000
error = 0
error1 = 0
i = 0
while i < maxloop:
    i = i + 1
    q += (y - h(x)) * numpy.matrix(x) * alpha
    error = h(x).sum()
    if abs(error - error1) < epsilon:
        break
    else:
        error1 = error
print(i)
matplotlib.pyplot.plot(x[numpy.where(y == 1),1],x[numpy.where(y == 1),2],'ro')
matplotlib.pyplot.plot(x[numpy.where(y == 0),1],x[numpy.where(y == 0),2],'bo')
matplotlib.pyplot.plot(x[:,1],-x[:,1] * q[:,1]/q[:,2] + -q[:,0] / q[:,2])
matplotlib.pyplot.show()


