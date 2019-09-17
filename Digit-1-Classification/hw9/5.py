import numpy as np
import matplotlib.pyplot as plt
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

def comb(d):
    coeff = []
    j = d
    for i in range(d+1):
        coeff.append((j,i))
        j -= 1
    return coeff

def get_coeff(d = 8):
    coeff = []
    for i in range(d+1):
        coeff += comb(i)
    return coeff

def hypothesis(w, x, y, coeff):
    result = 0
    for i in range(len(coeff)):
        result += w[i]*(x**coeff[i][0])*(y**coeff[i][1])
    return result

def nonlinear_t(x, y, coeff):
    pt = []
    for i in range(len(coeff)):
        pt.append((x**coeff[i][0])*(y**coeff[i][1]))
    return list(pt)

def classify():
    ones_x = []
    ones_y = []
    other_x = []
    other_y = []
    tones_x = []
    tones_y = []
    tother_x = []
    tother_y = []
    xs = []
    ys = []
    ptxm = []
    ptym = []
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
    f = open('new.train', 'r')
    i = 0
    for line in f:
        line = line.split(' ')
        y = getSymmetry(line[1:-1])
        x = getIntensity(line[1:-1])
        xs.append(x)
        ys.append(y)
        if line[0] == '1.0000':
            tones_y.append(y)
            tones_x.append(x)
        else:
            tother_y.append(y)
            tother_x.append(x)
        i += 1
    print(i)
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
    tones_x = np.array(tones_x)
    tones_x = 2 * (tones_x - xs.min())/xs.ptp() - 1
    tones_y = np.array(tones_y)
    tones_y = 2 * (tones_y - ys.min())/ys.ptp() - 1
    tother_x = np.array(tother_x)
    tother_x = 2 * (tother_x - xs.min())/xs.ptp() - 1
    tother_y = np.array(tother_y)
    tother_y = 2 * (tother_y - ys.min())/ys.ptp() - 1
    ones, = plt.plot(tones_x, tones_y, 'bo', label="Digit 1")
    others, = plt.plot(tother_x, tother_y, 'rx', label="Others")

    #load data
    #=====================================#
    coeff = get_coeff()
    for i in range(len(ones_x)):
        ptx = nonlinear_t(ones_x[i], ones_y[i], coeff)
        ptxm.append(ptx)
        ptym.append([1])
    for i in range(len(other_y)):
        ptx = nonlinear_t(other_x[i], other_y[i], coeff)
        ptxm.append(ptx)
        ptym.append([-1])

    x_matrix = np.matrix(ptxm)
    y_matrix = np.matrix(ptym)
    z = x_matrix.transpose().dot(x_matrix)
    _lambda = 0.071
    weight = np.linalg.inv(z + _lambda * np.eye(45)).dot(x_matrix.transpose()).dot(y_matrix)
    w = []
    for _w in weight:
        w.append(_w[0])
    w = np.array(w)

    #
    #=====================================#
    xr=np.linspace(-1,1)
    yr=np.linspace(-1,1)
    X, Y=np.meshgrid(xr, yr)
    Z = hypothesis(w, X,Y, coeff)
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    CS = plt.contour(xr, yr,Z,levels=[0], cmap="rainbow")
    CS.collections[0].set_label('g')
    plt.title("On test set $\lambda = 0.067$")
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    classify()
