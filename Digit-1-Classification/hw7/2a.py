import random
import numpy as np
import matplotlib.pyplot as plt

def gradient(x0, y0, eta):
    print(x0, y0, eta)
    x = x0
    y = y0
    iteration = 50
    fs = []
    it = []
    min_x = 0
    min_y = 0
    min_value = float("inf")
    for i in range(iteration):
        f = x**2 + 2*(y**2) + 2*np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
        if f < min_value:
            min_value = f
            min_x = x
            min_y = y
        x0 = eta*(2*x + 4*np.pi*np.cos(2*np.pi*x) * np.sin(2*np.pi*y))
        y0 = eta*(4*y + 4*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y))
        x -= x0
        y -= y0
        it.append(i)
        fs.append(f)
    print (x, y, f)
    print min_x, min_y, min_value
    plt.xlabel('iterations')
    plt.ylabel('f(x)')
    g, = plt.plot(it, fs, 'bo')
    plt.legend([g] , ['f(x)'])
    #plt.title("Gradient descent with $\eta=0.1$")
    #plt.grid()
    #plt.show()

if __name__ == "__main__":
    #gradient(0.1,0.1,0.01)
    #gradient(0.1,0.1,0.1)
    #gradient(1,1,0.01)
    #gradient(1,1,0.1)
    #gradient(-0.5,-0.5,0.01)
    #gradient(-0.5,-0.5,0.1)
    #gradient(-1,-1,0.01)
    gradient(-1,-1,0.1)

