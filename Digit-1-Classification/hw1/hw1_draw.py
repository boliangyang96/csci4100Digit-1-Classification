import matplotlib.pyplot as plt
import numpy as np
def f(x):
	return -2.0/3 * x - 1.0/3

t1 = np.arange(0.0, 5.0, 0.1)

plt.figure(1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(t1, f(t1))

plt.show()
