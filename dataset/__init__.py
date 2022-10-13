from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import random


class ChujianDataset(Dataset):
    '''
    Replacement for ImageFolder that supports empty subdirs.

    Args:
        root (string): Root directory path.
        transform (callable, optional): 
            A function/transform that takes in an PIL image.
        shuffle (bool, optional): Whether to shuffle the dataset.

    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    '''

    def __init__(
        self,
        root: str,
        transform=None,
        shuffle: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.shuffle = shuffle

        # Loop through root directory and get all classes and image paths.
        self.imgs = []
        self.classes = []
        self.class_to_idx = {}
        for glyph_dir in sorted(self.root.iterdir()):
            if not glyph_dir.is_dir():
                continue
            self.classes.append(glyph_dir.name)
            cls_idx = len(self.classes) - 1

            image_paths = sorted(glyph_dir.iterdir())
            
            # Always pick 1000 images from 
            class_size = max(100, len(image_paths))
            image_paths = random.choices(image_paths, k=class_size)
            for image_path in image_paths:
                self.imgs.append((image_path, cls_idx))
        if shuffle:
            random.shuffle(self.imgs)
        
        # Duplicate all images to make the dataset balanced.
        for idx in range(len(self.imgs)):
            self.imgs.append(self.imgs[idx])

    def __getitem__(self, idx: int) -> tuple:
        '''
        Return (image, class_index)
        '''
        image, label = self.imgs[idx]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.imgs)
