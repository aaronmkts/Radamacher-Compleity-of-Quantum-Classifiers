import numpy as np
import matplotlib.pyplot as plt

def gen_data(m, d):
    x=np.zeros((m,d))
    y = np.random.randint(0, 2, m)

    for i in range (m):
        mu = (-np.ones(d)) ** y[i]
        x[i,:]=np.random.normal(mu, np.ones(d))

    return x, y

"""
x, y = gen_data(1000, 2)
c=[]

for i in y:
    if (i==0):
       c.append('b')
    else:
       c.append('g')

plt.scatter(x[:, 0],x[:,1],c=c)
plt.show()
"""