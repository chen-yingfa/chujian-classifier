from pathlib import Path
from shutil import copyfile
import random


src_dir = Path('./chujian/glyphs')
all_images = []
for glyph_dir in src_dir.iterdir():
    for image in glyph_dir.iterdir():
        all_images.append(image)

dst_dir = Path('./chujian/glyphs_small')
images = random.sample(all_images, 1000)
for image in images:
    dst_file = dst_dir / image.parent.name / image.name
    print(dst_file)
    exit()
    dst_file.parent.mkdir(exist_ok=True, parents=True)
    copyfile(image, dst_file)
