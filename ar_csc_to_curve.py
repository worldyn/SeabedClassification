import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

#FILE = "x.aga.csv"
IN_DIR = "./stranton"
OUT_DIR = "./stranton-imgs"

ctr = 0
for filename in os.listdir(IN_DIR):
    print("{}: {}".format(str(ctr).zfill(4), filename))
    ctr += 1
    angles = []
    amplis = []
    path = '{}/{}'.format(IN_DIR, filename)
    reader = csv.reader(open(path, 'r'))

    for row in reader:
       angle = float(row[0].strip())
       ampli = float(row[1].strip())
       if ampli > 0.0001:
           angles.append(angle)
           amplis.append(ampli)

    plt.plot(angles, amplis, 'b-')
    plt.savefig('{}/{}.png'.format(OUT_DIR,filename))
    plt.clf()
