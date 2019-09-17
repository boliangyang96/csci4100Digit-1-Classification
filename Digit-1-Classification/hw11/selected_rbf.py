import numpy as np
import matplotlib.pyplot as plt
import math, operator
from rbf import rbf


# calculate symmetry about y-axis and x-axis
def getSymmetry(array):
    #"""
    val0 = 0
    for i in range(0,8):
        for j in range(0, 16):
            val0 += abs(float(array[j*16 + i]) - float(array[(j+1)*16 - (i + 1)]))
    val0 /=128
    #"""
    val1 = 0
    for i in range(0,16):
        for j in range(0,8):
            val1 += abs(float(array[j*16 + i]) - float(array[(15 - j)*16 - i]))
    val1 /=128
    return val1 + val0

def getIntensity(array):
    val = 0
    for i in range(0,256):
        val += abs(float(array[i]))
    return val

def train(k, xx, yy, data):
    ptxm = []
    ptym = []
    for i in data:
        ptx = [i[0], i[1]]
        ptxm.append(ptx)
        label = i[2]
        ptym.append(label)
    nn = rbf(k)
    ptxm = np.array(ptxm)
    ptym = np.array(ptym)
    #print(ptxm)
    #print(ptym)
    nn.train(ptxm, ptym)
    zz = []
    #print(nn.predict(ptxm))
    x_len, y_len = np.shape(xx)
    tptx = []
    for i in range(x_len):
        for j in range(y_len):
            px = xx[i][j]
            py = yy[i][j]
            ptx = [px, py]
            tptx.append(ptx)
    #print(tptx)
    z = nn.predict(np.array(tptx))
    #print(z)
    for i in range(x_len):
        z0 = []
        for j in range(y_len):
            x = z[i*x_len+j]
            if x < 0:
                z0.append(-1)
            else:
                z0.append(1)
        zz.append(z0)
    zz = np.array(zz)
    return zz

def classify():
    ones_x = []
    ones_y = []
    other_x = []
    other_y = []
    xs = []
    ys = []
    data = []
    #normalization
    #=====================================#

    #new.test
    #ZipDigits.combine
    #f = open('ZipDigits.combine', 'r')
    f = open('new.train', 'r')
    for line in f:
        line = line.split(' ')
        y = getSymmetry(line[1:-1])
        x = getIntensity(line[1:-1])
        xs.append(x)
        ys.append(y)
        if line[0] == '1.0000':
            ones_y.append(y)
            ones_x.append(x)
        else:
            other_y.append(y)
            other_x.append(x)
    f.close()
    f = open('new.test', 'r')
    for line in f:
        line = line.split(' ')
        y = getSymmetry(line[1:-1])
        x = getIntensity(line[1:-1])
        xs.append(x)
        ys.append(y)
    f.close()
    xs = np.array(xs)
    ys = np.array(ys)
    ones_x = np.array(ones_x)
    ones_x = 2 * (ones_x - xs.min())/xs.ptp() - 1
    ones_y = np.array(ones_y)
    ones_y = 2 * (ones_y - ys.min())/ys.ptp() - 1
    other_x = np.array(other_x)
    other_x = 2 * (other_x - xs.min())/xs.ptp() - 1
    other_y = np.array(other_y)
    other_y = 2 * (other_y - ys.min())/ys.ptp() - 1
    ones, = plt.plot(ones_x, ones_y, 'bo', markeredgecolor='k', label="Digit 1")
    others, = plt.plot(other_x, other_y, 'ro', markeredgecolor='k', label="Others")

    #load data
    #=====================================#
    for i in range(len(ones_x)):
        ptx = [ones_x[i], ones_y[i], 1]
        data.append(ptx)
    for i in range(len(other_y)):
        ptx = [other_x[i], other_y[i], -1]
        data.append(ptx)
    #k=5

    #
    #=====================================#
    X, Y=np.meshgrid(np.arange(-1, 1, 0.001),
                         np.arange(-1, 1, 0.001))
    Z = train(17, X, Y, data)
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    CS = plt.contourf(X, Y, Z, colors=('r', 'b'))
    plt.title("Decision boundary for $k = 17$")
    plt.legend(loc='upper right')
    #plt.grid()
    plt.show()

if __name__ == "__main__":
    classify()
