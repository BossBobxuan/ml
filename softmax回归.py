import numpy
import matplotlib.pyplot
import os 
dirpath = "./knn-digits/trainingDigits/"
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
x = numpy.ones((1934,1025))
x[:,1:1025] = numpy.array(finalarray)
y = numpy.array(y)
def qtx(x,q):
    return numpy.dot(q,x.T)
def pf(x,q):
    a = numpy.zeros((10,1934))
    for i in range(10):
        for j in range(10):
            a[i,:] += numpy.exp(qtx(x,q[j]))
        
    return  - numpy.exp(qtx(x,q)) / a
q = numpy.zeros((10,1025))
alpha = 0.001
epsilon = 0.0001
maxloop = 40000
error = numpy.zeros((1,10))
error1 = numpy.zeros((1,10))
j = 0
isend = numpy.zeros((1,10))
while j < maxloop:
    j += 1
    #sumarray = numpy.zeros((10,1025))
    for i in range(10):
        if isend[:,i] == 0:
            a = pf(x,q)
            a[i,y == i] += 1
            q[i,:] += alpha * numpy.dot(a[i,:],x)
            a[i,y == i] -= 1
            error[:,i] = - a[i,:].sum()
            print(error)
            if j == 1:
                error1[:,i] = error[:,i].copy()
            else:
                if abs(error[:,i] - error1[:,i]) < epsilon:
                    isend[:,i] = 1
                else:
                    error1[:,i] = error[:,i].copy()
    a = [e for e in isend[0] if e == 0]
    if len(a) == 0:
        break
print(j)
numpy.savetxt('softmaxend.txt',q)




