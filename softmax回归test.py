import numpy
import os
dirpath = "./knn-digits/testDigits/"
filename = os.listdir(dirpath)
finalarray = []
y = []
for name in filename[1:]:
    y.append(int(name[0]))
    with open(dirpath + name) as f:
        b = []
        for line in range(32):
            a = f.readline()
            for i in range(32):
                b.append(int(a[i]))
        finalarray.append(b) 
x = numpy.ones((946,1025))
x[:,1:1025] = numpy.array(finalarray)
y = numpy.array(y)
print(y)
q = numpy.loadtxt('softmaxend.txt')
print(q.shape)
def qtx(x,q):
    return numpy.dot(q,x.T)
def pf(x,q):
    a = numpy.zeros((10,946))
    for i in range(10):
        for j in range(10):
            a[i,:] += numpy.exp(qtx(x,q[j]))
        
    return  numpy.exp(qtx(x,q)) / a
results = qtx(x,q)
print(results[:,700])
a = results[:,700]
print(numpy.where(a == a.max()))
correct = 0
for i in range(946):
    a = results[:,i]
    #print(numpy.where(a == a.max())[0][0])
    if numpy.where(a == a.max())[0][0] == y[i]:
        correct += 1
print(correct / 946)

