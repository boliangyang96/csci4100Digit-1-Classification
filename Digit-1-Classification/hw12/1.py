import numpy as np
import matplotlib.pyplot as plt

def init():
    w0 = []
    w1 = []
    for i in range(3):
        w1.append(0.25)
    for i in range(2):
        w0.append(list(w1))
    w0 = np.array(w0)
    w1 = np.array(w1)
    return w0, w1



def nn1():
    #forward
    x1 = np.array([1, 1, 1])
    w0, w1 = init()
    x2 = np.tanh(np.dot(w0, x1))
    x2 = np.insert(x2, 0, 1)
    print("x2: ", x2)
    h = np.dot(w1, x2)
    print("h: ", h)
    #backprop
    d2 = 2 * (h - 1) / 4
    print("d2: ", d2)
    dw2 = np.dot(d2, x2)
    print("dw2: ", dw2)
    d1 = (1 - x2[1:] * x2[1:]) * np.dot(d2, w1[1:])
    print("d1: ", d1)
    dw1 = np.outer(x1, d1)
    print("dw1: ", dw1)
    print()
    print("numerical:")
    d = 0.0001
    #numerical
    dw2 = []
    for i in range(3):
        w1_p = np.array(w1)
        w1_p[i] += d
        w1_n = np.array(w1)
        w1_n[i] -= d
        w20 = ((1 - np.dot(w1_p, x2)) ** 2 - (1 - np.dot(w1_n, x2)) ** 2)/(2 * d * 4)
        dw2.append(w20)
    print("dw2: ", np.array(dw2))
    dw1 = []
    for i in range(2):
        tmp = []
        for j in range(3):
            w0_p = np.array(w0)
            w0_p[i][j] += d
            x2_p = np.tanh(np.dot(w0_p, x1))
            x2_p = np.insert(x2_p, 0, 1)
            h_p = np.dot(w1, x2_p)
            w0_n = np.array(w0)
            w0_n[i][j] -= d
            x2_n = np.tanh(np.dot(w0_n, x1))
            x2_n = np.insert(x2_n, 0, 1)
            h_n = np.dot(w1, x2_n)
            w10 = ((1 - h_p) ** 2 - (1 - h_n) ** 2)/(2 * d * 4)
            tmp.append(w10)
        dw1.append(tmp)
    print("dw1: ", np.array(dw1))

def nn2():
    #forward
    x1 = np.array([1, 1, 1])
    w0, w1 = init()
    x2 = np.tanh(np.dot(w0, x1))
    print("s1: ", np.dot(w0, x1))
    x2 = np.insert(x2, 0, 1)
    print("x1: ", x2)
    print("s2: ", np.dot(w1, x2))
    h = np.tanh(np.dot(w1, x2))
    print("h: ", h)
    #backprop
    d2 = 2 * (h - 1) * (1 - h ** 2) / 4
    print("d2: ", d2)
    dw2 = np.dot(d2, x2)
    print("dw2: ", dw2)
    print(x2 * x2)
    d1 =((1 - x2[1:] * x2[1:]) * np.dot(d2, w1[1:]))
    print("d1: ", d1)
    dw1 = np.outer(x1, d1)
    print("dw1: ", dw1)
    print()
    print("numerical:")
    d = 0.0001
    #numerical
    dw2 = []
    for i in range(3):
        w1_p = np.array(w1)
        w1_p[i] += d
        h_p = np.tanh(np.dot(w1_p, x2))
        w1_n = np.array(w1)
        w1_n[i] -= d
        h_n =  np.tanh(np.dot(w1_n, x2))
        w20 = ((1 - h_p) ** 2 - (1 - h_n) ** 2)/(2 * d * 4)
        dw2.append(w20)
    print("dw2: ", np.array(dw2))
    dw1 = []
    for i in range(2):
        tmp = []
        for j in range(3):
            w0_p = np.array(w0)
            w0_p[i][j] += d
            x2_p =  np.tanh(np.dot(w0_p, x1))
            x2_p = np.insert(x2_p, 0, 1)
            h_p =  np.tanh(np.dot(w1, x2_p))
            w0_n = np.array(w0)
            w0_n[i][j] -= d
            x2_n =  np.tanh(np.dot(w0_n, x1))
            x2_n = np.insert(x2_n, 0, 1)
            h_n =  np.tanh(np.dot(w1, x2_n))
            w10 = ((1 - h_p) ** 2 - (1 - h_n) ** 2)/(2 * d * 4)
            tmp.append(w10)
        dw1.append(tmp)
    print("dw1: ", np.array(dw1))

if __name__ == "__main__":
    print("identity:")
    nn1()
    print()
    print()
    print("tanh:")
    nn2()
