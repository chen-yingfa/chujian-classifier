from pathlib import Path

import torch
from torch import nn, Tensor
from torchvision import transforms

from dataset import ChujianDataset
from trainer import Trainer


def load_model() -> nn.Module:
    model = torch.hub.load(
        'pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    return model


def train(
    model,
    train_data: ChujianDataset,
):
    pass


def main():
    output_dir = Path('result/chujian_955')
    train_dir = Path('data/chujian/glyphs_955_train')
    test_dir = Path('data/chujian/glyphs_955_test')
    img_size = (50, 50)
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # TODO: Add more transformations: data augmentation, normalize etc.
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomResizedCrop(img_size),
        # transforms.RandomPosterize(bits=2),
        transforms.RandomAdjustSharpness(sharpness_factor=4),
        transforms.RandomInvert(),
        transforms.RandomAutocontrast(),
        transforms.RandomGrayscale(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_data = ChujianDataset(train_dir, train_transform, True)
    test_data = ChujianDataset(test_dir, test_transform, False)

    model = load_model()
    trainer = Trainer(model, output_dir, 2, 4, 0.001)
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main()
