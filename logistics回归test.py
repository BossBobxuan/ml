import numpy
import os
def qtx(x,q):
    return numpy.matrix(q) * numpy.matrix(x.T)
q = numpy.loadtxt("end.txt")
dirpath = "./knn-digits/testDigits/"
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
x = numpy.ones((946,1025))
x[:,1:1025] = numpy.array(finalarray)
print(x.shape) 

y = numpy.ones((1,x.shape[0]))
y[:,0:87] = numpy.zeros((1,87))
results = qtx(x,q)
correct = 0
for i in range(946):
    if results[:,i] > 0:
        results[:,i] = 1
    else:
        results[:,i] = 0
    if results[:,i] == y[:,i]:
        correct += 1
ratio = correct / 946
print(ratio)

