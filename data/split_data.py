from pathlib import Path
from shutil import copyfile
import random


# data_dir = Path('./chujian/glyphs_small')
# train_dir = Path('./chujian/glyphs_small_train')
# test_dir = Path('./chujian/glyphs_small_test')
data_dir = Path('./chujian/glyphs')
train_dir = Path('./chujian/glyphs_train')
test_dir = Path('./chujian/glyphs_test')

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
images = []
for glyph_dir in data_dir.iterdir():
    for image in glyph_dir.iterdir():
        images.append(image)

# Split images into train and test.
split_idx = int(len(images) * 0.8)
random.shuffle(images)
train_images = images[:split_idx]
test_images = images[split_idx:]

print('Train images:', len(train_images))
print('Test images:', len(test_images))

# Copy images to train and test.
print('Copying train images...')
for image in train_images:
    dst_file = train_dir / image.parent.name / image.name
    dst_file.parent.mkdir(exist_ok=True, parents=True)
    copyfile(image, dst_file)


print('Copying test images...')
for image in test_images:
    dst_file = test_dir / image.parent.name / image.name
    dst_file.parent.mkdir(exist_ok=True, parents=True)
    copyfile(image, dst_file)
