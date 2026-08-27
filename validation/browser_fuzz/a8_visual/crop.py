import sys
from PIL import Image
src, dst, box, scale = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
l, t, r, b = [int(v) for v in box.split(",")]
im = Image.open(src).crop((l, t, r, b))
im = im.resize((int(im.width * scale), int(im.height * scale)), Image.NEAREST)
im.save(dst)
print(dst, im.size)
