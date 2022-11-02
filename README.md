# Chujian Classifier

Implementation of Prototypical Network on classification of Chujian (楚简) characters. Support Prototypical Network using CNN, ResNet and ViT.

## Data

A small part (1,000 examples) of the entire Chujian dataset is extracted for convenience, it is located in `data/chujian/glyphs_small`. 

## Usage

### Training

Set the data path with `--train_dir` and `--test_dir`.
Set the output path with `--output_dir`.

Execute

```bash
python3 train_protonet.py
python3 train_resnet.py
python3 train_vit.py
```

### Testing

```bash
python3 train_protonet.py --mode test
python3 train_resnet.py --mode test
python3 train_vit.py --mode test
```

## Result

| Model     | Accuracy  |
| ---       | ---       |
| CNN       | <50       |
| ResNet-50 | 0.85      |
| ViT       | ?         |

> Note that ResNet and ViT are pretrained on ImageNet.

## Acknowledgements 

Much of the code is copied from THUNLP's [IsOBS](https://github.com/thunlp/IsOBS). 



