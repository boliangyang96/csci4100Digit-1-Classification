import numpy as np
import matplotlib.pyplot as plt
from knn import knn

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

def CV_test(data, xs, ys, data_test, k):
    ptxm = []
    ptym = []
    tptxm = []
    tptym = []
    for i in data:
        x = 2 * (i[0] - xs.min())/xs.ptp() - 1
        y = 2 * (i[1] - ys.min())/ys.ptp() - 1
        ptx = [x, y]
        ptxm.append(ptx)
        label = i[2]
        ptym.append(label)
    for i in data_test:
        x = 2 * (i[0] - xs.min())/xs.ptp() - 1
        y = 2 * (i[1] - ys.min())/ys.ptp() - 1
        ptx = [x, y]
        tptxm.append(ptx)
        label = i[2]
        tptym.append(label)
    nn = knn(k)
    cv_error = 0
    in_error = 0
    for i in range(len(data)):
        _ptxm = list(ptxm)
        _ptym = list(ptym)
        test_x = np.array([_ptxm.pop(i)])
        test_y = _ptym.pop(i)
        x_matrix = np.array(_ptxm)
        y_matrix = np.array(_ptym)
        nn.train(x_matrix, y_matrix)
        if nn.predict(test_x)[0] * test_y < 0:
            cv_error += 1
    #print cv_error
    cv_error /= float(len(data))
    ptxm = np.array(ptxm)
    ptym = np.array(ptym)
    nn.train(ptxm, ptym)
    tptxm = np.array(tptxm)
    tptym = np.array(tptym)
    y_test = nn.predict(tptxm)
    test_error = (np.multiply(y_test, tptym)<0).sum()/ float(len(tptym))
    y_test = nn.predict(ptxm)
    in_error = (np.multiply(y_test, ptym)<0).sum()/ float(len(tptym))
    return test_error, cv_error, in_error

def classify():
    test = []
    xs = []
    ys = []
    data = []
    data_test = []
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
            data.append((x,y,1))
        else:
            data.append((x,y,-1))
    xs = np.array(xs)
    ys = np.array(ys)
    f.close()

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

    #load data
    #=====================================#
    lambdas = []
    testes = []
    cves = []
    CV_min = 1000
    test_min = 1000
    i_min = 0
    k = 7
    print(CV_test(data, xs, ys, data_test, 7))
    """
    for i in range(150):
        _lambda = i * 2 + 1
        #print(_lambda)
        test_error, cv_error= CV_test(data, xs, ys, data_test, _lambda)
        lambdas.append(_lambda)
        testes.append(test_error)
        cves.append(cv_error)
        #print(i, test_error, cv_error)
        if CV_min>=cv_error:
            i_min = _lambda
            CV_min = cv_error
    print(i_min, CV_min)
    #
    #=====================================#
    f, = plt.plot(lambdas, cves, 'bo')
    #print(lambdas)
    plt.legend([f] , [r"$E_{cv}$"])
    plt.xlabel("$k$")
    plt.ylabel('Error')
    plt.title("Error versus k")
    plt.grid()
    plt.show()"""
    

if __name__ == "__main__":
    classify()
