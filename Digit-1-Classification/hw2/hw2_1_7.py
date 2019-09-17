import numpy as np
import matplotlib.pyplot as plt
from math import factorial
def nCr(n,r):
    f = factorial
    return f(n) // f(r) // f(n-r)
probabilities = []

x = np.arange(0, 1, 0.001);
for i in range(1000):
    probabilities.append(0)
for i in range(0, 7):
    for j in range(0, 7):
        p1 = .5**6 * nCr(6,i)
        p2 = .5**6 * nCr(6,j)
        probability = p1*p2
        div1 = abs(i / 6.0 - 0.5)
        div2 = abs(j / 6.0 - 0.5)
        div = max(div1, div2)
        #print(probability)
        k = 0
        while k/1000.0 < div and k < 1000:
            probabilities[k] += probability
            k += 1

plt.title("Probability vs Hoeffding bound")
plt.xlabel("$\epsilon$")
plt.ylabel('Probability')
f_line,  = plt.plot(x, probabilities, 'c')
g_line,  = plt.plot(x, 4*np.exp(-12*x**2), 'b')
plt.legend([f_line, g_line] , ['Probability estimation', 'Hoeffding bound'])
plt.grid(True)
plt.show()
