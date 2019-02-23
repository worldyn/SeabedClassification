#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv

FILE = "backscatter_data/RM30_Aug2013_10cm.csv"


reader = csv.reader(open(FILE, 'r'))
xs = []
ys = []

#ctr = 1000000

for row in reader:
#    if ctr == 0:
#        break
    #if ctr % 100 != 0:
    #    continue
    #ctr -= 1

    x = float(row[0].strip())
    y = float(row[1].strip())
    xs.append(x)
    ys.append(y)



fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(xs,ys,c='b')
ax.set_xlabel('x')
ax.set_ylabel('y')
#ax.set_xlim([-5,-2])
#ax.set_ylim([0,0.5])
#ax.set_zlim([-120,-60])

plt.savefig('shape.png')
#plt.show()


