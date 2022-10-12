import json
from pathlib import Path
from shutil import copyfile


src_dir = Path('/data/private/chenyingfa/chujian/data')
dst_dir = Path('/data/private/chenyingfa/chujian/glyphs')
meta_file = src_dir / 'chujian.json'
data = json.load(open(meta_file))

glyph_to_images = {}
num_images = 0
for ex in data:
    words = ex['sequences']
    images = ex['image_paths']
    images = [Path(image) for image in images]
    name_to_image = {f.name: f for f in images}
    for word in words:
        glyph = word['jian_word']
        if glyph is None:
            glyph = 'None'
        glyph = glyph.replace('/', ',')
        name = Path(word['jian_word_image']).name
        if name not in name_to_image:
            continue
        image = name_to_image[name]
        if glyph not in glyph_to_images:
            glyph_to_images[glyph] = []
        glyph_to_images[glyph].append(image)
        num_images += 1


print(f'Found {len(glyph_to_images)} glyphs')
print(f'Found {num_images} images')
print('------')

glyph_idx = 0
for glyph, images in glyph_to_images.items():
    glyph_idx += 1
    glyph_dir = dst_dir / glyph
    glyph_dir.mkdir(exist_ok=True, parents=True)
    print(f'{glyph_idx} Copying {len(images)} images for {glyph}')
    for image in images:
        filename = str(image)
        dst_filename = filename.replace('/', '_')
        dst_file = glyph_dir / dst_filename
        src_file = src_dir / image
        # print(src_file, dst_file)
        # exit()
        copyfile(src_file, dst_file)
