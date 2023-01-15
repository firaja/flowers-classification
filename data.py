import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2





l, u = 1e-5, 1e-3


x = [i*100 for i in range(10)]
y = [l if i%2==0 else u for i in range(10)]
y2 = [l, u, l, u/2, l, u/4, l, u/8, l, u/16]




fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
(ax1, ax2) = gs.subplots(sharex=True, sharey='row')



ax1.plot(x, y, label="sine")
ax1.set_title("Triangular")
ax2.set_title("Triangular2")
fig.text(0.5, 0.03, 'Training iterations', ha='center')
fig.text(0.05, 0.5, 'Learning rate', va='center', rotation='vertical')
ax2.plot(x, y2, color='orange', label="sine")

plt.show()
