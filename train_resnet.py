from pathlib import Path

import torch
from torch import nn, Tensor
from torchvision import transforms

from dataset import ChujianDataset
from trainer import Trainer
from modeling.resnet import ResNet


def load_model(img_size, num_classes) -> nn.Module:
    model = ResNet(img_size, num_classes)
    return model


def main():
    output_dir = Path('result/chujian_955')
    train_dir = Path('data/chujian/glyphs_955_train')
    test_dir = Path('data/chujian/glyphs_955_test')
    img_size = (50, 50)
    num_classes = 955
    
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

    print('Loading model...')
    model = load_model(img_size, num_classes)
    print('Instantiating trainer...')
    trainer = Trainer(model, output_dir, 2, 4, 0.001)
    trainer.train(train_data, test_data)


if __name__ == '__main__':
    main()
