import numpy as np
import matplotlib.pyplot as plt
import math, operator

neg_x = [1,0, 0,-1]
neg_y = [0,1,-1, 0]
pos_x = [0, 0,-2]
pos_y = [2,-2, 0]
x = []

def transform(x,y):
    nx = (x ** 2 + y ** 2) ** 0.5
    if x != 0:
        ny = np.arctan(y/ x)
    else:
        if y < 0:
            ny = -np.pi/2
        else:
            ny = np.pi/2
    return nx,ny

for i in range(len(neg_x)):
    #x.append((neg_x[i], neg_y[i], -1))
    nx, ny = transform(neg_x[i], neg_y[i])
    x.append((nx,ny, -1))
for i in range(len(pos_x)):
    #x.append((pos_x[i], pos_y[i], 1))
    nx, ny = transform(pos_x[i], pos_y[i])
    x.append((nx,ny, 1))
    
x = np.array(x)
x_min, x_max = -4, 4
y_min, y_max = -4, 4

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
def getNeighbors(px,py,k):
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


def train(k):
    zz = []
    x_len, y_len = np.shape(xx)
    for i in range(x_len):
        z0 = []
        for j in range(y_len):
            px = xx[i][j]
            py = yy[i][j]
            #z = getNeighbors(px, py, k)
            nx, ny = transform(px, py)
            z = getNeighbors(nx, ny, k)
            z0.append(z)
        zz.append(z0)
    zz = np.array(zz)
    return zz
#nx, ny = transform(2, 0)
#print(getNeighbors(nx, ny, 1))
zz = train(3)
plt.contourf(xx, yy, zz,colors=('r', 'b'))
plt.xlabel('x1')
plt.ylabel('x2')
pos_pts, = plt.plot(pos_x, pos_y, 'bo', markeredgecolor='k', label="+1")
neg_pts, = plt.plot(neg_x, neg_y, 'ro', markeredgecolor='k', label="-1")
plt.legend(loc='upper right')
#plt.xlim(-4, 4)
#plt.ylim(-4, 4)
#plt.grid()
plt.title("Decision regions for the 3-NN rule in the z-space")
#plt.title("3-NN with nonlinear transform")
plt.show()
