import sys

from PIL import Image
im = Image.open(sys.argv[1])

out = im.point(lambda i: int(i / 2)) # can actually use // (integer division)

outfile = "Q2.jpg"
try:
    out.save(outfile)
except IOError:
    print("cannot convert", infile)    