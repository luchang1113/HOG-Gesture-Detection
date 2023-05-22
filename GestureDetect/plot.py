import pandas as pd
out_a = pd.read_csv('./out_rate_A.csv',header=None)
out_v = pd.read_csv('./out_rate_V.csv',header=None)
out_five = pd.read_csv('./out_rate_Five.csv',header=None)
out_c = pd.read_csv('./out_rate_C.csv',header=None)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x_a = out_a[0]
y_a = out_a[1]
z_a = out_a[2]

x_v = out_v[0]
y_v = out_v[1]
z_v = out_v[2]

x_five = out_five[0]
y_five = out_five[1]
z_five = out_five[2]

x_c = out_c[0]
y_c = out_c[1]
z_c = out_c[2]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_a,y_a,z_a,)
ax.scatter(x_v,y_v,z_v,c='g')
ax.scatter(x_five,y_five,z_five,c='y')
ax.scatter(x_c,y_c,z_c,c='r')
plt.show()


