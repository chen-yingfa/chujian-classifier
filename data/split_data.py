from pathlib import Path
from shutil import copyfile
import random

from tqdm import tqdm


# data_dir = Path('./chujian/glyphs_small')
# train_dir = Path('./chujian/glyphs_small_train')
# test_dir = Path('./chujian/glyphs_small_test')

data_dir = Path('/data/private/chenyingfa/chujian/glyphs_955')

src_dir = data_dir / 'all'
train_dir = data_dir / 'train'
dev_dir = data_dir / 'dev'
test_dir = data_dir / 'test'

data_dir = Path('/data/private/chenyingfa/chujian/glyphs_merged_all/all')
train_dir = Path('/data/private/chenyingfa/chujian/glyphs_merged_all/train')
dev_dir = Path('/data/private/chenyingfa/chujian/glyphs_merged_all/dev')
test_dir = Path('/data/private/chenyingfa/chujian/glyphs_merged_all/test')

train_dir.mkdir(exist_ok=True, parents=True)
dev_dir.mkdir(exist_ok=True, parents=True)
test_dir.mkdir(exist_ok=True, parents=True)

# Make all glyph dirs, such that train and test have the same classes.
print('Creating glyph dirs...')
for glyph_dir in src_dir.iterdir():
    glyph_name = glyph_dir.name
    train_glyph_dir = train_dir / glyph_name
    dev_glyph_dir = dev_dir / glyph_name
    test_glyph_dir = test_dir / glyph_name
    train_glyph_dir.mkdir(exist_ok=True, parents=True)
    dev_glyph_dir.mkdir(exist_ok=True, parents=True)
    test_glyph_dir.mkdir(exist_ok=True, parents=True)

print('Looping image files...')
images = {}
for glyph_dir in src_dir.iterdir():
    glyph = glyph_dir.name
    images[glyph] = []
    for image in glyph_dir.iterdir():
        images[glyph].append(image)

# Split into train and test set
# For each class, split by 8:1:1 ratio.
print('Splitting...')
train_images = {}
dev_images = {}
test_images = {}
for glyph, image_files in images.items():
    random.seed(0)
    random.shuffle(image_files)
    # Floored to make sure test and dev has at least one example.
    split_idx = [
        int(len(image_files) * 0.8),
        int(len(image_files) * 0.9),
    ]
    train_images[glyph] = image_files[:split_idx[0]]
    dev_images[glyph] = image_files[split_idx[0] : split_idx[1]]
    test_images[glyph] = image_files[split_idx[1] : ]
    
def dict_size(d) -> int:
    return sum(len(v) for v in d.values())
print('Train images:', dict_size(train_images))
print('Dev images:', dict_size(dev_images))
print('Test images:', dict_size(test_images))

# Copy images 
def copy_images(img_dict, target_dir):
    for glyph, images in tqdm(img_dict.items()):
        for image in images:
            dst_file = target_dir / glyph / image.name
            dst_file.parent.mkdir(exist_ok=True, parents=True)
            copyfile(image, dst_file)


print('Copying train images...')
copy_images(train_images, train_dir)
print('Copying dev images...')
copy_images(dev_images, dev_dir)
print('Copying test images...')
copy_images(test_images, test_dir)
