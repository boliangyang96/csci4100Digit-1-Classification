import numpy as np
import matplotlib.pyplot as plt
import random
from random import shuffle

def generate_points(sep=5, rad=10, thk=5,  num=1000):
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

    shuffle(data)
    w = np.array([0, 0, 0])
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
    return iterations

if __name__ == "__main__":
    seps = []
    all_iterations = []

    sepv = 0.2
    while sepv <= 5.1:
        seps.append(sepv)
        all_iterations.append(generate_points(sep=sepv))
        sepv += 0.2
    pts, = plt.plot(seps, all_iterations, 'ro')
    plt.xlabel('sep')
    plt.ylabel('iterations')
    #plt.title("sep versus iterations")
    #plt.grid(True)
    plt.show()
