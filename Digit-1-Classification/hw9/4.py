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

def change_w(weight):
    w = []
    for _w in weight:
        w.append(_w[0])
    return np.array(w)

def CV_test(data, xs, ys, data_test, _lambda, coeff):
    ptxm = []
    ptym = []
    tptxm = []
    tptym = []
    for i in data:
        x = 2 * (i[0] - xs.min())/xs.ptp() - 1
        y = 2 * (i[1] - ys.min())/ys.ptp() - 1
        ptx = nonlinear_t(x, y, coeff)
        ptxm.append(ptx)
        label = i[2]
        ptym.append([label])
    for i in data_test:
        x = 2 * (i[0] - xs.min())/xs.ptp() - 1
        y = 2 * (i[1] - ys.min())/ys.ptp() - 1
        ptx = nonlinear_t(x, y, coeff)
        tptxm.append(ptx)
        label = i[2]
        tptym.append([label])
    #compute test error
    x_matrix = np.matrix(ptxm)
    y_matrix = np.matrix(ptym)
    z = x_matrix.transpose().dot(x_matrix)
    w = np.linalg.inv(z + _lambda * np.eye(45)).dot(x_matrix.transpose()).dot(y_matrix)
    #w = change_w(w.tolist())
    yh = np.inner(w.transpose(),np.matrix(tptxm))
    #yh = np.inner(tptxm, w).tolist()
    test_error = 0
    test_error = yh - np.array(tptym).transpose()
    test_error = test_error.dot(test_error.transpose())
    #print(np.shape(y_matrix))
    #for i in range(len(data_test)):
    #    test_error += (yh[i] -  tptym[i][0]) ** 2
    #print(np.shape(test_error))
    test_error = test_error/len(data_test)
    #compute Cross Validation Error
    cv_error = 0
    yh = np.inner(w.transpose(),x_matrix).tolist()[0]
    #print(np.shape(cv_error))
    y_matrix = y_matrix.tolist()
    h = x_matrix.dot(np.linalg.inv(z + _lambda * np.eye(45)).dot(x_matrix.transpose()))
    for i in range(len(data)):
        error = (yh[i] - y_matrix[i][0])/(1-h.item(i,i))
        cv_error += error**2
    cv_error = cv_error/(len(data))
    """
    for i in range(len(data)):
        _ptxm = list(ptxm)
        _ptym = list(ptym)
        test_x = _ptxm.pop(i)
        test_y = _ptym.pop(i)
        x_matrix = np.matrix(_ptxm)
        y_matrix = np.matrix(_ptym)
        z = x_matrix.transpose().dot(x_matrix)
        weight = np.linalg.inv(z + _lambda * np.eye(45)).dot(x_matrix.transpose()).dot(y_matrix)
        w = change_w(weight.tolist())
        h = x_matrix.dot(np.linalg.inv(z + _lambda * np.eye(45)).dot(x_matrix.transpose()))
        hnn = np.trace(h)
        yh = np.inner(test_x,w)
        if yh * test_y[0]<0:
            cv_error += 1/((1-hnn)**2)
    """
    test_error = test_error.tolist()[0][0]
    return test_error, cv_error

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
    coeff = get_coeff()
    CV_min = 1000
    test_min = 1000
    i_min = 0
    for i in range(201):
        _lambda = i  / 100.0
        test_error, cv_error = CV_test(data, xs, ys, data_test, _lambda, coeff)
        lambdas.append(_lambda)
        testes.append(test_error)
        cves.append(cv_error)
        #print(i, test_error, cv_error)
        if CV_min>=cv_error:
            i_min = _lambda
            CV_min = cv_error
            test_min = test_error
    print(i_min, CV_min, test_error)
    #
    #=====================================#
    g, = plt.plot(lambdas, testes, 'ro')
    f, = plt.plot(lambdas, cves, 'bo')
    plt.legend([g, f] , [r'$E_{test}$', r"$E_{cv}$"])
    plt.xlim(-0.01,2.01)
    plt.ylim(0.0, 0.15)
    plt.xlabel("$\lambda$")
    plt.ylabel('Error')
    #plt.title("Error versus $\lambda$")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    classify()
