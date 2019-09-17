import numpy as np
import matplotlib.pyplot as plt
from random import random, sample
from time import time
import operator
from matplotlib.pyplot import cm

def generate_a(num=10000):
    x = []
    for i in range(num):
        x.append((random(), random()))
    return list(x)

def generate_an(num=10000):
    centers = np.random.uniform(0,1,(10,2))
    sigma = 0.1 **2
    cov = sigma * np.eye(2)
    x = []
    for mean in centers:
        temp = np.random.multivariate_normal(mean, cov, 1000).tolist()
        x = x + temp
    return list(x), centers

def getNeighborsB(px, py, x):
    mind = 1000
    for ptx, pty in x:
        dist = (ptx - px) ** 2 + (pty - py) ** 2
        dist = dist ** 0.5
        if mind > dist:
            mind = dist
    return mind

def getNeighborsbb(px, py, center, new_x, rad, k1 = 10):
    x = []
    dists = []
    for k in range(k1):
        ptx, pty = center[k]
        dist = (ptx - px) ** 2 + (pty - py) ** 2
        dists.append((dist, k))
    dists.sort(key=operator.itemgetter(0))
    mdist, ms = dists.pop()
    x = new_x[ms]
    mdist = getNeighborsB(px, py, x)
    dists.sort(key=operator.itemgetter(1))
    for dist, k in dists:
        if mdist >= dist - rad[k]:
            x = new_x[ms]
            m_new = getNeighborsB(px, py, x)
            if m_new < mdist:
                mdist = m_new
    return mdist


def partition(x, k1 = 11):
    center = [0]
    rad = []

    for k in range(k1):
        maxd, maxk = 0, 0
        for i in range(len(x)):
            if i not in center:
                ptx, pty = x[i]
                mind = 1000
                for j in range(len(center)):
                    px, py = x[center[j]]
                    dist = (ptx - px) ** 2 + (pty - py) ** 2
                    if mind > dist:
                        mind = dist
                if maxd < mind:
                    maxd = mind
                    maxk = i
        center.append(maxk)

    new_x = []
    for i in range(k1):
        new_x.append([x[center[i]]])
        rad.append(0)

    for i in range(len(x)):
        if i not in center:
            ptx, pty = x[i]
            mind, minj = 1000, 0
            for j in range(k):
                px, py = x[center[j]][0], x[center[j]][1]
                dist = (ptx - px) ** 2 + (pty - py) ** 2
                if mind > dist:
                    mind = dist
                    minj = j
            new_x[minj].append((ptx, pty))
    index = 0
    centers = []
    for k in range(k1):
        cx, cy = 0, 0
        for ptx, pty in new_x[k]:
            cx += ptx
            cy += pty
        if len(new_x[k]) == 1:
            index = k
        cx /= len(new_x[k])
        cy /= len(new_x[k])
        centers.append((cx, cy))
        for ptx, pty in new_x[k]:
            dist = (ptx - cx) ** 2 + (pty - cy) ** 2
            if dist > rad[k]:
                rad[k] = dist
    centers.pop(index)
    new_x.pop(index)
    rad.pop(index)

    return centers, new_x, rad

if __name__ == "__main__":
    xx,_ = generate_an()
    colors = iter(cm.rainbow(np.linspace(0, 1, 11)))
    center, new_x, rad = partition(xx)
    for i in range(10):
        data = np.array(new_x[i])
        x, y = data.T
        plt.scatter(x, y, color=next(colors))
    c = np.array(center)
    x, y = c.T
    f = plt.scatter(x, y, color=next(colors))
    plt.legend([f] , [r"Partition Centers"])
    plt.title("Guassian partition")
    plt.grid()
    #plt.show()
    
    #x = generate_an()
    qx = generate_a()
    #center, new_x, rad = partition(x)
    print("finished partition")
    start_time = time()
    for px, py in qx:
        #getNeighborsB(px, py, xx)
        getNeighborsbb(px, py, center, new_x, rad)
    print (time()-start_time)
    
