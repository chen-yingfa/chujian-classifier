from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class ChujianDataset(Dataset):
    '''
    Replacement for ImageFolder that supports empty subdirs.
    
    Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    '''
    def __init__(
        self,
        root: str,
        transform=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform

        # Loop through root directory and get all classes and image paths.
        self.imgs = []
        self.classes = []
        self.class_to_idx = {}
        for glyph_dir in sorted(self.root.iterdir()):
            if not glyph_dir.is_dir():
                continue
            self.classes.append(glyph_dir.name)
            cls_idx = len(self.classes) - 1
            for image_path in glyph_dir.iterdir():
                self.imgs.append((image_path, cls_idx))

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
