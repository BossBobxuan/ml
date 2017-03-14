import numpy
import os
dirpath = "./knn-digits/trainingDigits/"
filename = os.listdir(dirpath)
finalarray = []
for name in filename[1:]:
    with open(dirpath + name) as f:
        b = []
        for line in range(32):
            a = f.readline()
            for i in range(32):
                b.append(int(a[i]))
        finalarray.append(b) 
x = numpy.array(finalarray)
y = numpy.ones((1,x.shape[0]))
y[:,0:189] = numpy.zeros((1,189))
def py0(x):
    results = numpy.zeros((1,1024))
    for i in range(189):
            results[0,:] += x[i,:]
    return results / 189
def py1(x):
    results = numpy.zeros((1,1024))
    for i in range(189,1934):
            results[0,:] += x[i,:]
    return results / (1934 - 189)
py = (1934 - 189) / 1934
p1 = py1(x)
p0 = py0(x)
testdirpath = "./knn-digits/testDigits/"
testfilename = os.listdir(testdirpath)
testfinalarray = []
for name in testfilename[1:]:
    with open(testdirpath + name) as f:
        b = []
        for line in range(32):
            a = f.readline()
            for i in range(32):
                b.append(int(a[i]))
        testfinalarray.append(b)
testx = numpy.array(finalarray)
testy = numpy.ones((1,testx.shape[0]))
testy[:,0:87] = numpy.zeros((1,87))
correct = 0
for j in range(946):
    a1 = 1
    a0 = 1
    for i in range(1024):
        if testx[j,i] == 1:
            a1 *= p1[:,i]
            a0 *= p0[:,i]
    p = a1 / (a1 * py + a0 * (1 - py))
    if p > 0.5:
        if testy[:,j] == 1:
            correct += 1
    else:
        if testy[:,j] == 0:
            correct += 1
print(correct / 946)

