import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

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

def hypothesis(slope, intercept, x):
    return x * slope + intercept


def classify():
    ones_x = []
    ones_y = []
    fives_x = []
    fives_y = []
    ptxm = []
    ptym = []
    data = []
    f = open('ZipDigits.test.txt', 'r')
    #f = open('ZipDigits.train', 'r')
    for line in f:
        line = line.split(' ')
        if line[0] == '1.0000':
            ones_y.append(getSymmetry(line[1:-1]))
            ones_x.append(getIntensity(line[1:-1]))
        elif line[0] == '5.0000':
            fives_y.append(getSymmetry(line[1:-1]))
            fives_x.append(getIntensity(line[1:-1]))
    f.close()
    f = open('ZipDigits.train', 'r')
    for line in f:
        line = line.split(' ')
        if line[0] == '1.0000':
            y_pos = getSymmetry(line[1:-1])
            x_pos = getIntensity(line[1:-1])
            ptx = np.array([1, x_pos, y_pos])
            ptxm.append(ptx)
            data.append((ptx, 1))
            ptym.append([1])
        elif line[0] == '5.0000':
            y_pos = getSymmetry(line[1:-1])
            x_pos = getIntensity(line[1:-1])
            ptx = np.array([1, x_pos, y_pos])
            data.append((ptx, -1))
            ptxm.append(ptx)
            ptym.append([-1])
    f.close()

    ones, = plt.plot(ones_x, ones_y, 'bo')
    fives, = plt.plot(fives_x, fives_y, 'rx')
    x_matrix = np.matrix(ptxm)
    y_matrix = np.matrix(ptym)
    w = np.linalg.pinv(x_matrix).dot(y_matrix).tolist()
    #pocket
    shuffle(data)
    weight = [w[0][0], w[1][0], w[2][0]]
    w = np.array(weight)

    i = 0
    yh = np.inner(ptxm, w).tolist()
    falsey = 0
    for h in yh:
        y = ptym[i]
        if h*y[0]<0:
            falsey += 1.0
        i += 1
    error0 = falsey
    iterations = 0
    flag = False
    while iterations<20000:
        flag = False
        for i in range(len(data)):
            ptx = data[i][0]
            y = data[i][1]
            if np.dot(w, ptx) * y <= 0:
                w1 = np.add(w, 0.13 * y * ptx)
                falsey = 0
                i = 0
                yh = np.inner(ptxm, w1).tolist()
                for h in yh:
                    y = ptym[i]
                    if h*y[0]<0:
                        falsey += 1.0
                    i += 1
                error = falsey
                if error < error0:
                    error0 = error
                    iterations += 1
                    flag = True
                    w = w1
        if flag == False:
            break

    g_slope = -1.0*w[1]/w[2]
    g_intercept = -1.0*w[0]/w[2]
    t = np.arange(0.79, 1, 0.01)
    print (g_slope, g_intercept)
    x_list = ones_x + fives_x
    y_list = ones_y + fives_y
    one = len(ones_x)
    falsey = 0
    i = 0
    for x_pos in x_list:
        h = hypothesis(g_slope, g_intercept, x_pos)
        if i < one and h < y_list[i]:
            falsey += 1.0
        elif i >= one and h > y_list[i]:
            falsey += 1.0
        i += 1
    print(float(i))
    print(falsey / float(i))
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    g_line, = plt.plot(t, t*g_slope + g_intercept, 'g')
    plt.legend([ones, fives, g_line] , ['1','5', 'g'])
    plt.title("On test set")
    #plt.grid()
    plt.show()

if __name__ == "__main__":
    classify()
