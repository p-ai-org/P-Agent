# Import packages
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx 
import numpy as np

# Generate heatmap data
pack_point_x = 530 #531
pack_point_y = 110
pack_point_z = 42
num = 20

pos_x = np.zeros((num,num,num))
pos_y = np.zeros((num,num,num))
pos_z = np.zeros((num,num,num))
# 3d points around initial conditions
for i in range(-int(num/2),int(num/2)):
    for j in range(-int(num/2),int(num/2)):
        for k in range(0,int(num)):
            pos_x[i,j,k] = pack_point_x + i
            pos_y[i,j,k] = pack_point_y + j
            pos_z[i,j,k] = pack_point_z + k

dist = np.zeros((num,num,num))
# Rel. distance vector for num points
for i in range(-int(num/2),int(num/2)):
    for j in range(-int(num/2),int(num/2)):
        for k in range(0,int(num)):
            x = pos_x[i,j,k]
            y = pos_y[i,j,k]
            z = pos_z[i,j,k]
            dist[i,j,k] = np.sqrt(( x - pack_point_x)**2 + (y - pack_point_y)**2 + (z - pack_point_z)**2)

# Create figure and add axis
cm = plt.get_cmap('jet')        # Set the colorsmap by fetching from matplotlib.pyplot and store in variable
dist = dist.flatten()
cNorm = matplotlib.colors.Normalize(vmin=min(dist), vmax=max(dist))         # Normalized range of the colormap based on the values in the 3 space
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)         # Compiles scalarmap object with normalized range and colormap

fig = plt.figure()          # Create figure object
ax = Axes3D(fig)            # Returns a 3d axes object for the parent figure
ax.scatter(pos_x, pos_y, pos_z, c=scalarMap.to_rgba(np.transpose(dist)))            #converts array of colormap values to COLORS and plots out all points in 3space

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



plt.show()