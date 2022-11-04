import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('plots/benchmarkTime.csv', delimiter=',').astype(int)
uElements = np.unique(data[:,0]).astype(int)

bar_width = 0.35
index = np.arange(6)
fig= plt.subplots()
    a = data[np.where(data[:,0] == i)]
    b = a[:,2].astype(int)
    plt.bar(index, b, bar_width, alpha=0.8 )
    index = index+bar_width
plt.xticks(uElements)
plt.legend(uElements, loc='upper left')
plt.xlabel('Upscale Size')
plt.ylabel('Number of cycles') 
plt.title('Benchmark Implementation Performance')
plt.savefig("./plots/result%d.png"%i)
plt.close()


# import matplotlib as mpl
# mpl.use('Agg')
# import numpy as np
# import matplotlib.pyplot as plt

# # data to plot
# n_groups = 4
# means_frank = (90, 55, 40, 65)
# means_guido = (85, 62, 54, 20)

# # create plot
# fig, ax = plt.subplots()
# index = np.arange(n_groups)
# bar_width = 0.35
# opacity = 0.8

# rects1 = plt.bar(index, means_frank, bar_width,
# alpha=opacity,
# color='b',
# label='Frank')

# rects2 = plt.bar(index + bar_width, means_guido, bar_width,
# alpha=opacity,
# color='g',
# label='Guido')

# plt.xlabel('Person')
# plt.ylabel('Scores')
# plt.title('Scores by person')
# plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
# plt.legend()

# plt.tight_layout()
# plt.savefig("./plots/result.png")