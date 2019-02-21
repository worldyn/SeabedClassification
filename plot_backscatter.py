from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv

FILE = "RM30_Aug2013_10cm.csv"


reader = csv.reader(open(FILE, 'r'))
d = {}
for i in range(5):
    inside = {'depth':[],'std':[],'intensity':[]}
    d[i+1] = inside

ctr = 1000
for row in reader:
  # k, v = row
   #print(row)
   depth = float(row[2].strip())
   std = float(row[3].strip())
   intensity = float(row[5].strip())
   cat = int(float(row[7].strip()))
   d[cat]['depth'].append(depth)
   d[cat]['std'].append(std)
   d[cat]['intensity'].append(intensity)
   ctr -= 1
   if ctr == 0:
       break

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

col = ['r','b','k','g','y']
for i in range(5):
    ax.scatter(d[i+1]['depth'],d[i+1]['std'],d[i+1]['intensity'],c=col[i])
ax.set_xlabel('depth')
ax.set_ylabel('std')
ax.set_zlabel('intensity')
ax.set_xlim([-5,-2])
ax.set_ylim([0,0.5])
ax.set_zlim([-120,-60])

plt.show()

