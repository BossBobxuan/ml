import numpy
import matplotlib.pyplot
data = numpy.loadtxt("ex0.txt")
mx = data[:,0:2]
print(mx)
my = data[:,2]
print(my)
mmx = numpy.array([(1, 0., 3), (1, 1., 3), (1, 2., 3), (1, 3., 2), (1, 4., 4)])
mmy = numpy.array([95.364, 97.217205, 75.195834, 60.105519, 49.342380])
def h(mx,q):
    return numpy.matrix(q) * numpy.matrix(mx).T
alpha = 0.001#一定要注意学习系数，值过大会错过收敛
epsilon = 0.0001
q = numpy.zeros((1,2))
error = 0.0
error1 = 0.0
num = 0
while True:
    num += 1
    q += (my - h(mx,q)) * numpy.matrix(mx) * alpha 
    error = (numpy.array((my - h(mx,q))) ** 2 / 2).sum()
    matplotlib.pyplot.plot(mx[:,1],(q[:,0] + q[:,1] * mx[:,1]))
    if abs(error - error1) < epsilon:
        break
    else:
        error1 = error
print(num)
matplotlib.pyplot.plot(mx[:,1],my,'ro')#画散点图否则点与点之间会被连线
matplotlib.pyplot.plot(mx[:,1],(q[:,0] + q[:,1] * mx[:,1]))
matplotlib.pyplot.show()
print(q)


