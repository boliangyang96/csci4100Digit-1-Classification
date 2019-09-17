import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle

num = 1000
def classify_print(num):
    slope = random.uniform(-1, 1)
    y_intercept = random.uniform(10, 20)
    print (slope, y_intercept)
    num_neg = int(num / 2) + random.randint(-num/4, num/4)
    data = []
    neg_x = []
    neg_y = []
    pos_x = []
    pos_y = []

    for i in range(0, num_neg):
        x = random.uniform(1, 49)
        y = x*slope + y_intercept
        y_pos = 51
        while y_pos >= 50:
            y_pos = random.uniform(1, y)
        ptx = np.array([1, x, y_pos])
        data.append((ptx, -1))
        neg_x.append(x)
        neg_y.append(y_pos)


    for i in range(num_neg, num):
        x = random.uniform(1, 49)
        y = x*slope + y_intercept
        y_pos = random.uniform(y, 49)
        ptx = np.array([1, x, y_pos])
        data.append((ptx, 1))
        pos_x.append(x)
        pos_y.append(y_pos)

    shuffle(data)

    plt.xlabel('x1')
    plt.ylabel('x2')

    t = np.arange(0.0, 50.0, 1)
    f_line, = plt.plot(t, t*slope + y_intercept, 'y')
    neg_pts, = plt.plot(neg_x, neg_y, 'bo')
    pos_pts, = plt.plot(pos_x, pos_y, 'ro')
    plt.axis([0, 50, 0, 50])
    #plt.legend([neg_pts, pos_pts, f_line] , ['-1','+1', 'f'])
    #plt.show()

    #1.4 (c)(d)(e)

    w = np.array([1, 0, 0])
    iterations = 0
    flag = False
    while True:
        flag = False
        for i in range(len(data)):
            ptx = data[i][0]
            y = data[i][1]
            if np.dot(w, ptx) * y <= 0:
                w = np.add(w, y * ptx)
                iterations += 1
                flag = True
        if flag == False:
            break
    print (iterations)
    g_slope = -1.0*w[1]/w[2]
    g_intercept = -1.0*w[0]/w[2]
    print (g_slope, g_intercept)
    #f_line, = plt.plot(t, t*slope + y_intercept, 'y')
    #neg_pts, = plt.plot(neg_x, neg_y, 'bo')
    #pos_pts, = plt.plot(pos_x, pos_y, 'ro')
    #plt.axis([0, 50, 0, 50])
    #plt.xlabel('x1')
    #plt.ylabel('x2')
    g_line, = plt.plot(t, t*g_slope + g_intercept, 'g')
    plt.legend([neg_pts, pos_pts, g_line, f_line] , ['-1','+1', 'g', 'f'])
    plt.show()
if __name__ == "__main__":

    #classify_print(20)
    #classify_print(100)
    classify_print(1000)

