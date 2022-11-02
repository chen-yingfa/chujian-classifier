from pathlib import Path
from argparse import ArgumentParser, Namespace
import json
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms

from dataset import ChujianDataset
from trainer import Trainer
from modeling.vit import Vit
from utils import get_param_cnt


def load_model(
    model_name: str,
    img_size: Tuple[int, int],
    num_classes: int,
    pretrained: bool
) -> nn.Module:
    model = Vit(model_name, img_size, num_classes, pretrained=pretrained)
    return model


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=16)
    p.add_argument("--mode", default="train_test")
    p.add_argument("--output_dir", default="result/glyphs_955")
    p.add_argument("--pretrained", type=bool, default=True)
    p.add_argument("--model_name", default="vit_base_patch16_224_in21k")
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


def main():
    assert torch.cuda.is_available(), "CUDA is not available"

    args = parse_args()
    print(json.dumps(args.__dict__, indent=4))

    train_dir = Path("./data/chujian/glyphs_955/train")
    dev_dir = Path("./data/chujian/glyphs_955/dev")
    test_dir = Path("./data/chujian/glyphs_955/test")
    # train_dir = Path("/data/private/chenyingfa/chujian/glyphs_955/train")
    # dev_dir = Path("/data/private/chenyingfa/chujian/glyphs_955/dev")
    # test_dir = Path("/data/private/chenyingfa/chujian/glyphs_955/test")
    img_size = (224, 224)
    num_classes = 955

    output_dir = Path(
        args.output_dir,
        args.model_name,
        f"lr{args.lr}-bs{args.batch_size}-ep{args.num_epochs}",
    )

    print("Loading model...", flush=True)
    model = load_model(args.model_name, img_size, num_classes, args.pretrained)
    print(f"Params: {get_param_cnt(model)}")

    train_transform = transforms.Compose(
        [
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
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    print("Instantiating trainer...", flush=True)
    trainer = Trainer(
        model,
        output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        lr=args.lr,
    )

    if "train" in args.mode:
        train_data = ChujianDataset(train_dir, train_transform, True)
        dev_data = ChujianDataset(dev_dir, test_transform, False)
        trainer.train(train_data, dev_data)
    if "test" in args.mode:
        test_data = ChujianDataset(test_dir, test_transform, False)
        test_output_dir = output_dir / "test"
        result = trainer.evaluate(test_data, test_output_dir)
        del result["preds"]
        print(result)


if __name__ == "__main__":
    main()