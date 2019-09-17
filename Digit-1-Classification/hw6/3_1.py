import numpy as np
import matplotlib.pyplot as plt
import random
from random import shuffle

def generate_points(rad=10, thk=5, sep=5, num=1000):
    data = []
    neg_x = []
    neg_y = []
    pos_x = []
    pos_y = []
    ptxm = []
    ptym = []
    neg_center = [0,sep/2]
    pos_center = [thk/2+rad,-sep/2]
    for i in range(num): # generate pts below line
        angle = random.uniform(0,np.pi)
        n_rad = rad + random.uniform(0,thk)
        x_pos = neg_center[0] - n_rad*np.cos(angle)
        y_pos = neg_center[1] + n_rad*np.sin(angle)
        ptx = np.array([1, x_pos, y_pos])
        ptxm.append(ptx)
        ptym.append([-1])
        data.append((ptx, -1))
        neg_x.append(x_pos)
        neg_y.append(y_pos)

    for i in range(num): # generate pts above line
        angle = random.uniform(0,np.pi)
        n_rad = rad + random.uniform(0,thk)
        x_pos = pos_center[0] - n_rad*np.cos(angle)
        y_pos = pos_center[1] - n_rad*np.sin(angle)
        ptx = np.array([1, x_pos, y_pos])
        ptxm.append(ptx)
        ptym.append([1])
        data.append((ptx, 1))
        pos_x.append(x_pos)
        pos_y.append(y_pos)

    plt.xlabel('x1')
    plt.ylabel('x2')
    shuffle(data)

    neg_pts, = plt.plot(neg_x, neg_y, 'ro')
    pos_pts, = plt.plot(pos_x, pos_y, 'bo')
    
    w = np.array([1, 1, 100])
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
    #print (iterations)
    #g_slope = -1.0*w[1]/w[2]
    #g_intercept = -1.0*w[0]/w[2]
    #t = np.arange(-(thk+rad+2), (3*thk/2+rad*2+2), 1)
    #g_line, = plt.plot(t, t*g_slope + g_intercept, 'g')
    #print (g_slope, g_intercept)
    #plt.legend([neg_pts, pos_pts, g_line] , ['-1','+1', 'g'])
    #plt.show()
    
    x_matrix = np.matrix(ptxm)
    y_matrix = np.matrix(ptym)
    w = np.linalg.pinv(x_matrix).dot(y_matrix).tolist()
    g_slope = -1.0*w[1][0]/w[2][0]
    g_intercept = -1.0*w[0][0]/w[2][0]
    print (g_slope, g_intercept)
    t = np.arange(-(thk+rad+2), (3*thk/2+rad*2+2), 1)
    g_line, = plt.plot(t, t*g_slope + g_intercept, 'g')
    plt.legend([neg_pts, pos_pts, g_line] , ['-1','+1', 'g'])
    #plt.title("Linear regression on double semi-circle data")
    #plt.grid()
    plt.show()

if __name__ == "__main__":
    generate_points()
