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
"""
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

def hypothesis(w, x, y):
    coeff = get_coeff(3)
    result = 0
    for i in range(len(coeff)):
        result += w[i]*(x**coeff[i][0])*(y**coeff[i][1])
    return result
"""
def hypothesis(w, x, y):
    return (w[0] + w[1]*x + w[2]*y + w[3]*(x ** 2) + w[4]*x*y + w[5]*(y**2) + w[6]*(x**3) + w[7]*(x**2)*y
    + w[8]*(y**2)*x + w[9]*(y**3))
def classify():
    ones_x = []
    ones_y = []
    fives_x = []
    fives_y = []
    ptxm = []
    ptym = []
    ptxm1 = []
    ptym1 = []
    data = []
    f = open('ZipDigits.test.txt', 'r')
    #f = open('ZipDigits.train', 'r')
    for line in f:
        line = line.split(' ')
        if line[0] == '1.0000':
            y = getSymmetry(line[1:-1])
            x = getIntensity(line[1:-1])
            ones_y.append(getSymmetry(line[1:-1]))
            ones_x.append(getIntensity(line[1:-1]))
            ptx = np.array([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3])
            ptxm1.append(ptx)
            ptym1.append([1])
        elif line[0] == '5.0000':
            y = getSymmetry(line[1:-1])
            x = getIntensity(line[1:-1])
            fives_y.append(getSymmetry(line[1:-1]))
            fives_x.append(getIntensity(line[1:-1]))
            ptx = np.array([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3])
            ptxm1.append(ptx)
            ptym1.append([-1])
    f.close()
    f = open('ZipDigits.train', 'r')
    for line in f:
        line = line.split(' ')
        if line[0] == '1.0000':
            y = getSymmetry(line[1:-1])
            x = getIntensity(line[1:-1])
            ptx = np.array([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3])
            data.append((ptx, 1))
            ptxm.append(ptx)
            ptym.append([1])
        elif line[0] == '5.0000':
            y = getSymmetry(line[1:-1])
            x = getIntensity(line[1:-1])
            ptx = np.array([1, x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3])
            data.append((ptx, -1))
            ptxm.append(ptx)
            ptym.append([-1])
    f.close()

    ones, = plt.plot(ones_x, ones_y, 'bo', label="1")
    fives, = plt.plot(fives_x, fives_y, 'rx', label="5")
    x_matrix = np.matrix(ptxm)
    y_matrix = np.matrix(ptym)
    weight = np.linalg.pinv(x_matrix).dot(y_matrix)
    w = weight.tolist()
    weight = [w[0][0], w[1][0], w[2][0], w[3][0], w[4][0], w[5][0], w[6][0], w[7][0], w[8][0], w[9][0]]
    w = np.array(weight)
    #pocket
    shuffle(data)
    falsey = 0
    i = 0
    yh = np.inner(ptxm,w).tolist()
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
                w1 = np.add(w, 0.045*y * ptx)
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


    xr=np.linspace(0.76,1)
    yr=np.linspace(0,1)
    #hypothesis=lambda x, y: (w[0] + w[1]*x + w[2]*y + w[3]*(x ** 2) + w[4]*x*y + w[5]*(y**2) + w[6]*(x**3) + w[7]*(x**2)*y
    #    + w[8]*(y**2)*x + w[9]*(y**3))
    X, Y=np.meshgrid(xr, yr)
    Z = hypothesis(w, X,Y)
    falsey = 0
    i = 0
    print(w)
    yh = np.inner(ptxm1,w).tolist()
    for h in yh:
        y = ptym1[i]
        if h*y[0]<0:
            falsey += 1.0
        i += 1
    print(falsey)
    print(falsey / float(i))
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    CS = plt.contour(xr, yr,Z,levels=[0], cmap="rainbow")
    CS.collections[0].set_label('g')
    #plt.legend([ones, fives] , ['1','5'])
    plt.legend(loc='upper right')
    plt.title("On test set")
    #plt.grid()
    plt.show()

if __name__ == "__main__":
    classify()
