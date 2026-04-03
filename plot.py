import matplotlib
matplotlib.use('TkAgg')
import os, sys
sys.path.insert(0, '/app')
from experiments.pybullet.plot_scalability2 import plot_line_data
plot_line_data()
import matplotlib.pyplot as plt
plt.savefig('/app/res/plots/scalability.png', dpi=150, bbox_inches='tight')
print("График сохранён в res/plots/scalability.png")
