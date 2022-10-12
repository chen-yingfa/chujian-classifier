from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torch import nn
from torchvision import transforms

from dataset import ChujianDataset
from trainer import Trainer
from modeling.resnet import ResNet
from utils import get_param_cnt


def load_model(img_size, num_classes) -> nn.Module:
    model = ResNet(img_size, num_classes)
    return model


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--mode', default='train_test')
    p.add_argument('--output_dir', default='result/temp')
    return p.parse_args()


def main():
    assert torch.cuda.is_available(), 'CUDA is not available'
    
    args = parse_args()
    
    train_dir = Path('/data/private/chenyingfa/chujian/glyphs_955_train')
    test_dir = Path('/data/private/chenyingfa/chujian/glyphs_955_test')
    img_size = (50, 50)
    num_classes = 955
    
    lr = args.lr
    output_dir = Path(args.output_dir)
    
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

    print('Loading model...', flush=True)
    model = load_model(img_size, num_classes)
    print(f'Params: {get_param_cnt(model)}')
    print('Instantiating trainer...', flush=True)
    trainer = Trainer(
        model, 
        output_dir,
        batch_size=args.batch_size,
        log_interval=50,
        lr=args.lr,
    )
    
    if 'train' in args.mode:
        trainer.train(train_data, test_data)
    if 'test' in args.mode:
        test_output_dir = output_dir / 'test'
        result = trainer.evaluate(
            test_data, test_output_dir)
        del result['preds']
        print(result)

if __name__ == '__main__':
    main()
