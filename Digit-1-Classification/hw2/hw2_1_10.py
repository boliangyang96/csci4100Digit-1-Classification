import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
def flip_coin():
    head = 0
    for i in range(10):
        head += random.randint(0,1)
    return head

def simulator():
    #0 for v0, 1 for random, 2 for min
    sol = []
    randI = random.randint(0, 999)
    min_freq = 1.0
    for i in range(1000):
        out = flip_coin()
        if (out < min_freq):
            min_freq = out
        sol.append(out)
    return (sol[0], sol[randI], min_freq)

if __name__ == "__main__":
    numtest = 100000
    v1_list={}
    vrand_list={}
    vmin_list={}
    v1_list = defaultdict(lambda:0,v1_list)
    vrand_list = defaultdict(lambda:0,vrand_list)
    vmin_list = defaultdict(lambda:0,vmin_list)

    bins_ =np.arange(0, 1.1, 0.1)
    for i in range(numtest):
        v1, vrand, vmin = simulator()
        #v1_list.append(v1)
        #vrand_list.append(vrand)
        #vmin_list.append(vmin)
        v1_list[v1] += 1
        vrand_list[vrand] += 1
        vmin_list[vmin] += 1

    p1 = []
    prand = []
    pmin = []
    #epsilon
    for i in range(11):
        r=[]
        #\nu
        for j in range(11):
            if abs(j-5)>i:
                r.append(j)
        t1 = 0
        t2 = 0
        t3 = 0
        for index in r:
            t1 += v1_list[index]
            t2 += vrand_list[index]
            t3 += vmin_list[index]
        n = 100000.0
        p1.append(t1/n)
        prand.append(t2/n)
        pmin.append(t3/n)

    x = np.arange(0, 1.1, 0.001)
    plt.subplot(3, 1, 1)
    plt.title("Probability estimation of $\\nu_1$, $\\nu_{rand}$, and $\\nu_{min}$ vs Hoeffding bound")
    #plt.hist(v1_list, bins=bins_)
    p,=plt.plot(x, 2*np.exp(-2*(x)**2 * 10))
    h,=plt.plot(bins_, p1, 'g')
    plt.legend([p,h], ['Estimation', 'Hoeffding bound'])
    plt.grid(True)
    plt.ylabel("$\\nu_1$")

    plt.subplot(3, 1, 2)
    #plt.hist(vrand_list, bins=bins_)
    p,=plt.plot(x, 2*np.exp(-2*(x)**2 * 10))
    h,=plt.plot(bins_, prand, 'g')
    plt.legend([p,h], ['Estimation', 'Hoeffding bound'])
    plt.grid(True)
    plt.ylabel("$\\nu_{rand}$")

    plt.subplot(3, 1, 3)
    #plt.hist(vmin_list, bins=bins_)
    p,=plt.plot(x, 2*np.exp(-2*(x)**2 * 10))
    h,=plt.plot(bins_, pmin, 'g')
    plt.legend([p,h], ['Estimation', 'Hoeffding bound'])
    plt.grid(True)
    plt.ylabel("$\\nu_{min}$")
    plt.xlabel("$\epsilon$")
    plt.show()
