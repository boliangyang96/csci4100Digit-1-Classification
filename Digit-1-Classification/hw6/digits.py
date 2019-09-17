import numpy as np
import matplotlib.pyplot as plt
f = open('ZipDigits.test.txt', 'r')

ones_x = []
ones_y = []
fives_x = []
fives_y = []

# calculate symmetry about y-axis and x-axis
def getSymmetry(array):
    val0 = 0
    val1 = 0
    for i in range(0,8):
        for j in range(0, 16):
            val0 += abs(float(array[j*16 + i]) - float(array[(j+1)*16 - (i + 1)]))
    val0/=128
    for i in range(0,16):
        for j in range(0,8):
            val1 += abs(float(array[j*16 + i]) - float(array[(15 - j)*16 - i]))
    val1/=128
    return val0*val1

def getIntensity(array):
    val = 0
    for i in range(0,256):
        val += abs(float(array[i]))
    return val / 256

for line in f:
    line = line.split(' ')
    if line[0] == '1.0000':
        ones_y.append(getSymmetry(line[1:-1]))
        ones_x.append(getIntensity(line[1:-1]))
    elif line[0] == '5.0000':
        fives_y.append(getSymmetry(line[1:-1]))
        fives_x.append(getIntensity(line[1:-1]))

ones, = plt.plot(ones_x, ones_y, 'bo', markerfacecolor='none')
fives, = plt.plot(fives_x, fives_y, 'rx')
plt.xlabel('Intensity')
plt.ylabel('Symmetry')
plt.legend([ones, fives] , ['1','5'])
#plt.title("Symmetry versus Intensity on test set")
#plt.grid()
plt.show()
