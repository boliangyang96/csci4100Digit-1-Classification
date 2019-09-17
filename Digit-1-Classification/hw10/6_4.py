import numpy as np
import matplotlib.pyplot as plt
import operator, random

def getNeighbors(px, py, k, x):
    distances = []
    for ptx, pty, tag in x:
        dist = (ptx - px) ** 2 + (pty - py) ** 2
        dist = dist ** 0.5
        distances.append((dist,tag))
    distances.sort(key=operator.itemgetter(0))
    result = 0
    for i in range(k):
        result += distances[i][1]
    if result < 0:
        return -1
    else:
        return 1

def train(xx, yy, k, x):
    zz = []
    x_len, y_len = np.shape(xx)
    for i in range(x_len):
        z0 = []
        for j in range(y_len):
            px = xx[i][j]
            py = yy[i][j]
            z = getNeighbors(px, py, k, x)
            z0.append(z)
        zz.append(z0)
    zz = np.array(zz)
    return zz

def generate_points(rad=10, thk=5, sep=5, num=1000):
    neg_x = []
    neg_y = []
    pos_x = []
    pos_y = []
    x = []
    neg_center = [0,sep/2]
    pos_center = [thk/2+rad,-sep/2]
    for i in range(num): # generate pts below line
        angle = random.uniform(0,np.pi)
        n_rad = rad + random.uniform(0,thk)
        x_pos = neg_center[0] - n_rad*np.cos(angle)
        y_pos = neg_center[1] + n_rad*np.sin(angle)
        x.append((x_pos, y_pos,1))
        neg_x.append(x_pos)
        neg_y.append(y_pos)

    for i in range(num): # generate pts above line
        angle = random.uniform(0,np.pi)
        n_rad = rad + random.uniform(0,thk)
        x_pos = pos_center[0] - n_rad*np.cos(angle)
        y_pos = pos_center[1] - n_rad*np.sin(angle)
        x.append((x_pos, y_pos,-1))
        pos_x.append(x_pos)
        pos_y.append(y_pos)

    plt.xlabel('x1')
    plt.ylabel('x2')
    x_min, x_max = -20, 30
    y_min, y_max = -20, 20

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))
    zz = train(xx, yy, 3, x)
    neg_pts, = plt.plot(neg_x, neg_y, 'ro', markeredgecolor='k', label="-1")
    pos_pts, = plt.plot(pos_x, pos_y, 'bo', markeredgecolor='k', label="+1")
    plt.contourf(xx, yy, zz,colors=('b', 'r'))
    plt.legend(loc='upper right')
    plt.title("Decision regions for the 3-NN rule")
    #plt.grid()
    plt.show()

if __name__ == "__main__":
    generate_points()
