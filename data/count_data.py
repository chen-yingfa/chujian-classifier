from pathlib import Path
import matplotlib.pyplot as plt

data_dir = Path('./chujian/glyphs_small_train')
# data_dir = Path('./chujian/glyphs')
# data_dir = Path('/data/private/chenyingfa/chujian/glyphs')
glyph_cnts = {}
for glyph_dir in data_dir.iterdir():
    glyph_cnts[glyph_dir.name] = len(list(glyph_dir.iterdir()))

cnts = list(glyph_cnts.items())
cnts = sorted(cnts, key=lambda x: x[1], reverse=True)
for i in range(10):
    print(cnts[i][0], cnts[i][1])

cnts = [cnt for _, cnt in cnts]
plt.plot(cnts)
plt.xlabel('Glyphs')
plt.ylabel('Number of images')
plt.savefig('cnts.png')
