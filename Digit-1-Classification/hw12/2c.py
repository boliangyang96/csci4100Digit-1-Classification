import numpy as np
import matplotlib.pyplot as plt
from nn import nn
from random import shuffle

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
    ones, = plt.plot(ones_x, ones_y, 'bo', markeredgecolor='w', label="Digit 1")
    others, = plt.plot(other_x, other_y, 'ro', markeredgecolor='k', label="Others")
    #load data
    #=====================================#
    for i in range(len(ones_x)):
        ptx = [ones_x[i], ones_y[i], 1]
        data.append(ptx)
    for i in range(len(other_y)):
        ptx = [other_x[i], other_y[i], -1]
        data.append(ptx)
    shuffle(data)
    X = []
    Y = []
    traind = data[:250]
    testd = data[250:]
    for d in traind:
        X.append([d[0], d[1]])
        Y.append([d[2]])

    X = np.array(X)
    Y = np.array(Y)

    Xt = []
    Yt = []
    for d in testd:
        Xt.append([d[0], d[1]])
        Yt.append([d[2]])
    ein = []
    ecv = []
    iteration = []
    nerualnetwork = nn()
    for i in range(50):#2000000
        iteration.append(i + 1)
        epsilon = 0.0004 / np.sqrt(i + 1)
        nerualnetwork.train(X, Y, epsilon)
        e1 = nerualnetwork.perror(X, Y)
        #print("e1: ", e1)
        ein.append(e1)
        e1 = nerualnetwork.perror(Xt, Yt)
        ecv.append(e1)
    xt = []
    yt = []
    data_test = []
    f = open('new.test', 'r')
    for line in f:
        line = line.split(' ')
        y = getSymmetry(line[1:-1])
        x = getIntensity(line[1:-1])
        if line[0] == '1.0000':
            data_test.append((x,y,1))
        else:
            data_test.append((x,y,-1))
    f.close()
    for i in data_test:
        x = 2 * (i[0] - xs.min())/xs.ptp() - 1
        y = 2 * (i[1] - ys.min())/ys.ptp() - 1
        ptx = [x, y]
        xt.append(ptx)
        label = i[2]
        yt.append([label])
    Xt = np.array(xt)
    Yt = np.array(yt)
    e1 = nerualnetwork.perror(Xt, Yt)
    print("test_error: ", e1)

    X, Y=np.meshgrid(np.arange(-1, 1, 0.001),
                         np.arange(-1, 1, 0.001))
    x_len, y_len = np.shape(X)
    tptx = []
    for i in range(x_len):
        for j in range(y_len):
            px = X[i][j]
            py = Y[i][j]
            ptx = [px, py]
            tptx.append(ptx)
    z = nerualnetwork.predict(np.array(tptx))
    Z = np.reshape(z, (x_len, y_len))
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    CS = plt.contourf(X, Y, Z, colors=('r', 'b'))
    plt.title("Decision boundary")
    plt.legend(loc='upper right')
    """
    f, = plt.plot(iteration, ein, 'bo')
    g, = plt.plot(iteration, ecv, 'ro')
    plt.legend([f, g] , [r"$E_{in}$", r"$E_{cv}$"])
    plt.xlabel("$iterations$")
    plt.ylabel('Error')
    plt.title("Error versus iterations")
    """
    #plt.grid()
    plt.show()


if __name__ == "__main__":
    classify()
