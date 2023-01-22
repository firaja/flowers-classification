import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker


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


x = [i*0.1 for i in range(11)]
y1 = [0.9492156600952148, 0.9411764740943909, 0.9225490093231201, 0.9343137145042419, 0.9323529601097107, 0.9323529601097107, 0.9264705777168274, 0.929411768913269, 0.9196078181266785, 0.9127451181411743, 0.820588231086731]
y2 = [18519982, 18494310, 18436494, 18269548, 17784182, 17115458, 15942038, 14702558, 11329546, 7407102, 971366]




fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
l1 = ax1.plot(x, y1, 'g-', label="Accuracy")
l2 = ax2.plot(x, y2, 'b-', label="Parameters")

ax1.set_xlabel('Freeze rate')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Parameters')
ax1.grid()
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlim([0, 1])

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : str(x/10e5)+"x${}$".format(f.format_data(1000000))
#ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
#ax2.set_ylim(971000, 18520000)

ax1.legend(l1+l2, ["Accuracy", "Trainable parameters"])
plt.show()
