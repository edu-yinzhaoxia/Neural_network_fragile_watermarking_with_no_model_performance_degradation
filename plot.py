import matplotlib.pyplot as plt
import numpy as np

correct = np.load('./correct.npy')
correct = correct/100.
print(correct)
y = np.arange(0,len(correct))
# plot
fig, ax = plt.subplots()
ax.plot(y,correct , linewidth=2.0)
ax.set_ylim(0, 1)
ax.set_xlim(0,len(correct))
plt.show()