from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class ChujianDataset(Dataset):
    '''Replacement for ImageFolder that supports empty subdirs.'''
    def __init__(
        self,
        root: str,
        transform=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform

        self.images = []
        self.classes = []
        for i, glyph_dir in enumerate(self.root.iterdir()):
            for image in glyph_dir.iterdir():
                self.images.append(image)
                self.classes.append(i)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.classes[idx]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)
