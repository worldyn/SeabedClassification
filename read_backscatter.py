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
    1: (0, 255, 255, 0), # red
    2: (255, 0, 255, 0), # green
    3: (255, 255, 0, 0), # blue
    4: (0, 0, 0, 0),     # white
    5: (0, 0, 255, 0)    # yellow
}

RANGES = {
    'd': (-21.940, -1.500),
    's': (0.000, 9.143),
    'a': (0.000, 1.950),
    'u': (-467.609, -18.322)
}

SETTINGS = {}
for k in RANGES:
    lo = float(RANGES[k][0])
    hi = float(RANGES[k][1])
    SETTINGS[k] = {'lo': lo, 'hi': hi}

# Translates feature with range in settings of value x to a color channel
def feat2color(settings, x):
    lo = settings['lo']
    hi = settings['hi']
    width = hi - lo
    if x == np.inf:
        return np.uint8(255)
    if width < 0.0001:
        return np.uint8(0)
    return np.uint8(((x-lo)/width) * 255)

# Width and height of a data point
#M = 10 # 1 sq meter
M = 512

class Pixel:
    def __init__(self, d, s, a, u, x, y, c):
        self.d = d
        self.s = s
        self.a = a
        self.u = u

        # only when reading raw from csv, don't export
        self.x = x
        self.y = y
        self.c = c

    def __repr__(self):
        return '(([{},{},{}],{},{},{},{}))'.format(
                self.x, self.y, self.c,
                self.d, self.s, self.a, self.u)

    def to_cmyk(self, settings=None):
        if settings is None:
            settings = SETTINGS
        # Arbitrary orders give different looking images
        out = (
            #feat2color(SETTINGS['d'], self.d),
            #feat2color(SETTINGS['s'], self.s),
            #feat2color(SETTINGS['a'], self.a),
            #feat2color(SETTINGS['u'], self.u)

            feat2color(SETTINGS['d'], self.d),
            feat2color(SETTINGS['a'], self.s),
            feat2color(SETTINGS['s'], self.a),
            feat2color(SETTINGS['u'], self.u)
        )
        return out

    def to_class_cmyk(self):
        return CLASS_CMYK[self.c]

def pix2cmyk(pix):
    return pix.to_cmyk()
pixels2cmyk_data = np.vectorize(pix2cmyk)

class Point:
    def __init__(self, pixels):
        self.pixels = pixels # should be M*M
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for (_,pix) in np.ndenumerate(pixels):
            counts[pix.c] += 1
        
        high_count = counts[1]
        high_class = 1
        for c in range(2,6):
            if counts[c] > high_count:
                high_count = counts[c]
                high_class = c

        self.c = high_class
        self.counts = counts
    
    def __str__(self):
        return 'class = {},\ncounts={}\n{}\n=='.format(
                self.c, self.counts, self.pixels)

    def to_img(self):
        cmyk_data = np.zeros((M,M,4), dtype=np.uint8)
        for i in range(M):
            for j in range(M):
                cmyk_data[i,j] = self.pixels[i][j].to_cmyk()
        img = Image.fromarray(cmyk_data, mode='CMYK')
        return img.convert(mode='RGB')
        #return img

    def to_img_high_contrast(self):
        pix = self.pixels[0][0]
        settings = {
            'd': {'lo': pix.d, 'hi': pix.d},
            's': {'lo': pix.s, 'hi': pix.s},
            'a': {'lo': pix.a, 'hi': pix.a},
            'u': {'lo': pix.u, 'hi': pix.u}
        }

        for row in self.pixels:
            for pix in row:
                settings['d']['lo'] = min(settings['d']['lo'], pix.d)
                settings['d']['hi'] = max(settings['d']['hi'], pix.d)
                settings['s']['lo'] = min(settings['s']['lo'], pix.s)
                settings['s']['hi'] = max(settings['s']['hi'], pix.s)
                settings['a']['lo'] = min(settings['a']['lo'], pix.a)
                settings['a']['hi'] = max(settings['a']['hi'], pix.a)
                settings['u']['lo'] = min(settings['u']['lo'], pix.u)
                settings['u']['hi'] = max(settings['u']['hi'], pix.u)

        cmyk_data = np.zeros((M,M,4), dtype=np.uint8)
        for i in range(M):
            for j in range(M):
                cmyk_pix = self.pixels[i][j].to_cmyk(settings=settings)
                cmyk_data[i,j] = cmyk_pix
        #return Image.fromarray(cmyk_data, mode='CMYK')
        return Image.fromarray(cmyk_data, mode='CMYK').convert(mode='RGB')
                    


    def to_class_img(self):
        cmyk_data = np.empty((M,M,4),dtype=np.uint8)
        for i in range(M):
            for j in range(M):
                for (k,col) in enumerate(self.pixels[i][j].to_class_cmyk()):
                    cmyk_data[i,j,k] = col
        img = Image.fromarray(cmyk_data, mode='CMYK')
        return img.convert(mode='RGB')


reader = csv.reader(open(FILE, 'r'))
points = []

def gen_pixels():
    for row in reader:
       d = float(row[2].strip())
       s = float(row[3].strip())
       a = float(row[4].strip())
       u = float(row[6].strip())
       x = float(row[0].strip())
       y = float(row[1].strip())
       c = int(float(row[7].strip()))
       pix = Pixel(d, s, a, u, x, y, c)
       yield pix

def gen_rows():
    pixels = gen_pixels()
    cur_row = [next(pixels)]
    cur_y = cur_row[0].y

    for pix in pixels:
        if pix.y == cur_y:
            cur_row.append(pix)
        else:
            yield cur_row
            cur_row = [pix]
            cur_y = pix.y
    yield cur_row

def gen_points():
    row_buf = []
    for row in gen_rows():
        row_buf.append(row)
        if len(row_buf) == M:
            for point in points_from_rows(row_buf):
                yield point
            row_buf = []


def points_from_rows(rows):
    min_len = min([len(r) for r in rows])

    if min_len < M:
        return []

    num_points = int(min_len / M)

    pixel_mats = [ np.empty((M,M), dtype=Pixel) for _ in range(num_points) ]

    for (i, row) in enumerate(rows):
        for (j,pix) in enumerate(row):
            if j >= M*num_points:
                break
            pixel_mats[int(j/M)][i,j%M] = pix

    return [Point(pixel_mat) for pixel_mat in pixel_mats]

# Num of points (pictures) to generate
K = 1
for (i,p) in enumerate(gen_points()):
    if i == K:
        break
    img = p.to_img()
    #img = p.to_img_high_contrast()
    num_str = str(i).zfill(4)
    img.save("X{}.png".format(num_str))
    p.to_class_img().save("Y{}.png".format(num_str))
    print("{} done".format(i))
