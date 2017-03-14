import numpy
import matplotlib.pyplot
import os
dirpath = "./knn-digits/trainingDigits/"
filename = os.listdir(dirpath)
print(filename)
finalarray = []
for name in filename[1:]:
    with open(dirpath + name) as f:
        b = []
        for line in range(32):
            a = f.readline()
            for i in range(32):
                b.append(int(a[i]))
        finalarray.append(b) 
x = numpy.ones((1934,1025))
x[:,1:1025] = numpy.array(finalarray)
y = numpy.ones((1,x.shape[0]))
y[:,0:189] = numpy.zeros((1,189))
q = numpy.zeros((1,1025))
def qtx(x,q):
    return numpy.matrix(q) * numpy.matrix(x.T)
def sigmoidfunc(mx):
    return 1 / (1 + numpy.exp(-mx))
def h(x):
    return sigmoidfunc(qtx(x,q))
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
numpy.savetxt("end.txt",q)



