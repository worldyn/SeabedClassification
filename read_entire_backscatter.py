import sys
from PIL import Image
import numpy as np
import csv

FILE = "backscatter_data/RM30_Aug2013_10cm.csv"

# d = depth
# s = stdev
# a = acoustic beam footprint
# u = unfiltered backscatter strength
# c = class (sand or whatnot) (but as a int)

CLASS_CMYK = {
    1: (0, 255, 255, 0), # red    = sand
    2: (255, 0, 255, 0), # green  = sand/gravel
    3: (255, 255, 0, 0), # blue   = gravel
    4: (0, 0, 0, 255),   # black  = sand/boulder/rock
    5: (0, 0, 255, 0)    # yellow = boulder/rock
}

RANGES = {
    'd': (-21.940, -1.500),
    's': (0.000, 9.143),
    'a': (0.000, 1.950),
    'u': (-467.609, -18.322)
}

#XMIN = 219761.930
#XMAX = 220293.830
#YMIN = 612354.274
#YMAX = 613013.774
# In decimeters
XMIN = 2197619
XMAX = 2202938
YMIN = 6123542
YMAX = 6130137

W = 5320
H = 6596

SETTINGS = {}
for k in RANGES:
    lo = float(RANGES[k][0])
    hi = float(RANGES[k][1])
    SETTINGS[k] = {'lo': lo, 'hi': hi}

SETTINGS['d']['exp'] = 1.0
SETTINGS['s']['exp'] = 0.1 # higher resolution aaround low values
SETTINGS['a']['exp'] = 0.2 # higher resolution aaround low values
SETTINGS['u']['exp'] = 2.0 # higher resolution aaround high values

def xstr2col(x):
    return int(float(x)*10) - XMIN

def ystr2row(y):
    return int(float(y)*10) - YMIN

# Translates feature with range in settings of value x to a color channel
def raw2cmyk(settings, x):
    lo = settings['lo']
    hi = settings['hi']
    width = hi - lo
    if x == np.inf:
        return np.uint8(255)
    if width < 0.0001:
        return np.uint8(0)
    return np.uint8(((x-lo)/width) * 255)

def raws2cmyk(d, s, a, u):
    # Different orders give different looking images
    return (
        raw2cmyk(SETTINGS['d'], d),
        raw2cmyk(SETTINGS['s'], s),
        raw2cmyk(SETTINGS['a'], a),
        raw2cmyk(SETTINGS['u'], u)
    )

def cat2cmyk(c):
    return CLASS_CMYK[c]

def raw2purple(name, x):
    lo = SETTINGS[name]['lo']
    hi = SETTINGS[name]['hi']
    width = hi - lo
    red = None
    if x == np.inf:
        red = np.uint8(255)
    elif width < 0.0001:
        red = np.uint8(0)
    else:
        #red = np.uint8(((x-lo)/width) * 255)
        norm = (x-lo) / width
        norm = norm ** SETTINGS[name]['exp']
        red = np.uint8(norm * 255)
    return (red, np.uint8(0), np.uint8(255))




feats = np.zeros((H,W,4), dtype=np.uint8)
cats = np.zeros((H,W,4), dtype=np.uint8)
ds = np.zeros((H,W,3), dtype=np.uint8)
ss = np.zeros((H,W,3), dtype=np.uint8)
acs = np.zeros((H,W,3), dtype=np.uint8)
us = np.zeros((H,W,3), dtype=np.uint8)


reader = csv.reader(open(FILE, 'r'))
for line in reader:
    i = ystr2row(line[1].strip())
    j = xstr2col(line[0].strip())

    d = float(line[2].strip())
    s = float(line[3].strip())
    a = float(line[4].strip())
    u = float(line[6].strip())
    c = int(float(line[7].strip()))

    feats[i,j] = raws2cmyk(d, s, a, u)
    cats[i,j] = cat2cmyk(c)
    ds[i,j] = raw2purple('d', d)
    ss[i,j] = raw2purple('s', s)
    acs[i,j] = raw2purple('a', a)
    us[i,j] = raw2purple('u', u)

feat_img = Image.fromarray(feats, mode='CMYK')
cat_img = Image.fromarray(cats, mode='CMYK')
ds_img = Image.fromarray(ds, mode='RGB')
ss_img = Image.fromarray(ss, mode='RGB')
as_img = Image.fromarray(acs, mode='RGB')
us_img = Image.fromarray(us, mode='RGB')

feat_img.save('feat.tiff', compression='tiff_deflate')
cat_img.save('cat.tiff', compression='tiff_deflate')
ds_img.save('ds.tiff', compression='tiff_deflate')
ss_img.save('ss.tiff', compression='tiff_deflate')
as_img.save('as.tiff', compression='tiff_deflate')
us_img.save('us.tiff', compression='tiff_deflate')
