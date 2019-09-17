import math
import numpy as np
import matplotlib.pyplot as plt
import random

def generatePoint():
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    slope = x1+x2
    y = -x1*x2
    return x1, x2, slope, y
def g(x, slope, y):
    return x*slope+y

def runExperiment():
    results = []
    gbarslope = 0
    gbary = 0
    point = []
    #n = int(input())
    n = 1000
    for i in range(0, n):
        x1, x2, slope, y = generatePoint()
        gbarslope += slope
        gbary += y
        results.append((slope,y))
        point.append((x1,x2))
    gbarslope /= n
    gbary /= n
    print ("gbarslope:%f gbary:%f" % (gbarslope, gbary))
    bias = 0
    var = 0
    for i in range(0, n):
        x1, x2 = point[i]
        y1 = x1**2
        y2 = x2**2
        gby1 = g(x1,gbarslope,gbary)
        gby2 = g(x2,gbarslope,gbary)
        bias += ((y1-gby1)**2+(y2-gby2)**2)/2
        tmp = 0
        for j in range(0,n):
            slope, y = results[j]
            tmp += (g(x1,slope,y)-gby1)**2+(g(x2,slope,y)-gby2)**2
        tmp/=2*n
        var+=tmp
    bias /= n
    var /= n
    print ("bias:", bias)
    print ("variance:", var)
    return gbarslope, gbary

results = runExperiment()
x = np.arange(-1, 1, 0.001)
plt.title(r"plot of $\bar{g}(x)$ and f(x)")
plt.xlabel("$x$")
plt.ylabel('y')
g_line, = plt.plot(x, x*results[0] + results[1], 'r', label=r'$\bar{g}(x)$')
f_line, = plt.plot(x, x*x, 'b', label='f(x)')
plt.legend(handles=[f_line, g_line])
plt.show()
