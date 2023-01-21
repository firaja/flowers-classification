import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

"""
Total params: 18,646,213
Trainable params: 18,519,982

0.1
Total params: 18,646,213
Trainable params: 18,494,310

0.2
Total params: 18,646,213
Trainable params: 18,436,494

0.3
Total params: 18,646,213
Trainable params: 18,269,548

0.4
Trainable params: 17,784,182

0.5
Trainable params: 17,115,458

0.6
Trainable params: 15,942,038

0.7
Trainable params: 14,702,558

0.8
Trainable params: 11,329,546

0.9
Trainable params: 7,407,102

1.0
Trainable params: 971,366


"""



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
