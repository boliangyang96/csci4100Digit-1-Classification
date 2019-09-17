import numpy as np
import matplotlib.pyplot as plt

def f1(t):
    return t ** 3
t1 = np.arange(-1, 1.1, 0.1)
f, = plt.plot([0]*len(t1), t1, 'b')
g, = plt.plot(t1, f1(t1), 'r')
plt.legend([f,g] , ["decision boundary for part (a)", "decision boundary for part (b)"])
#plt.grid()
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.title(r"Decision boundary")
plt.show()
