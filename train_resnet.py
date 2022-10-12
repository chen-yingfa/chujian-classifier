from pathlib import Path

import torch
from torch import nn, Tensor

from data import ChujianDataset


def load_model() -> nn.Module:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
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
    
    dataset = ChujianDataset(output_dir / 'glyphs_955_train')
    model = load_model()
    train(model, dataset)


if __name__ == '__main__':
    main()
