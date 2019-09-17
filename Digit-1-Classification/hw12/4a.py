import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import LeaveOneOut

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
    X = []
    Y = []
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
        ptx = [ones_x[i], ones_y[i]]
        Y.append(1)
        X.append(ptx)
    for i in range(len(other_y)):
        ptx = [other_x[i], other_y[i]]
        Y.append(-1)
        X.append(ptx)
    X = np.array(X)
    Y = np.array(Y)

    model = svm.SVC(kernel='poly', degree=8, coef0 =1,C=0.1)
    model.fit(X,Y)
    py = model.predict(X)
    error = len(Y) - np.sum(py == Y)
    print("Ein: ", float(error)/len(Y))

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
    Z = model.predict(tptx)
    Z = Z.reshape(X.shape)
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    CS = plt.contourf(X, Y, Z, colors=('r', 'b'))
    plt.title("Decision boundary for $C = 0.1$")
    plt.legend(loc='upper right')
    #plt.grid()
    plt.show()
#"""

if __name__ == "__main__":
    classify()
