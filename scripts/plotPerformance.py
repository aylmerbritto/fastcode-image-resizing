import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('performance.csv', delimiter=',')
data[:, 3] -= data[:, 2]
a = data[np.where(data[:,0] == 1.02400000e+03)]
b = data[np.where(data[:,0] == 2.04800000e+03)]

fig= plt.figure()
plt.plot(a[:,1], a[:,3], color ="red")
plt.plot(a[:,1], b[:,3], color ="green")
plt.savefig("./result.png")
plt.close()
print(a)
print(b)