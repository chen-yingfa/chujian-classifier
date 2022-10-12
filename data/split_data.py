from pathlib import Path
from shutil import copyfile
import random


# data_dir = Path('./chujian/glyphs_small')
# train_dir = Path('./chujian/glyphs_small_train')
# test_dir = Path('./chujian/glyphs_small_test')
data_dir = Path('./chujian/glyphs_955')
train_dir = Path('./chujian/glyphs_955_train')
test_dir = Path('./chujian/glyphs_955_test')

train_dir.mkdir(exist_ok=True, parents=True)
test_dir.mkdir(exist_ok=True, parents=True)

# Make all glyph dirs, such that train and test have the same classes.
print('Creating glyph dirs...')
for glyph_dir in data_dir.iterdir():
    glyph_name = glyph_dir.name
    train_glyph_dir = train_dir / glyph_name
    test_glyph_dir = test_dir / glyph_name
    train_glyph_dir.mkdir(exist_ok=True, parents=True)
    test_glyph_dir.mkdir(exist_ok=True, parents=True)

print('Looping image files...')
images = {}
for glyph_dir in data_dir.iterdir():
    glyph = glyph_dir.name
    images[glyph] = []
    for image in glyph_dir.iterdir():
        images[glyph].append(image)

# Split into train and test set
# For each class, split by 4:1 ratio.
train_images = {}
test_images = {}
for glyph, image_files in images.items():
    random.seed(0)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_images[glyph] = image_files[:split_idx]
    test_images[glyph] = image_files[split_idx:]

print('Train images:', len(train_images))
print('Test images:', len(test_images))

# Copy images to train and test.
print('Copying train images...')
for glyph, images in train_images.items():
    for image in images:
        dst_file = train_dir / glyph / image.name
        dst_file.parent.mkdir(exist_ok=True, parents=True)
        copyfile(image, dst_file)


print('Copying test images...')

for glyph, images in test_images.items():
    for image in images:
        dst_file = test_dir / glyph / image.name
        dst_file.parent.mkdir(exist_ok=True, parents=True)
        copyfile(image, dst_file)
