import numpy
from PIL import Image
im = Image.open("pic2.png")
im = im.resize((32,32))
def get_char(r,g,b,alpha = 256):
    if alpha == 0:
        return " "
    
    return int(0.3 * r + 0.59 * g + 0.11 * b)
txt = []
final = []
for i in range(32):
        for j in range(32):
            txt.append(get_char(*im.getpixel((j,i))))
for a in txt:
    if a < 200:
        final.append(1)
    else:
        final.append(0)
finalarray = numpy.ones((1,1025))
finalarray[:,1:1025] = numpy.array(final)
print(finalarray)
q = numpy.loadtxt("softmaxend.txt")
print(q)
results = numpy.dot(q,finalarray.T)
print(results)
print(numpy.where(results == results.max())[0][0])


