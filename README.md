# Chujian Classifier

Implementation of Prototypical Network on classification of Chujian (楚简) characters. Support Prototypical Network (based on CNN), ResNet and ViT.

## Data

Preprocess the data into specific format with the code in `data` folder. See `data/README.md` for details.

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

| Model     | Top-1 Acc.    | Top-5 Acc.   | Top-10 Acc. |
| ---       | ---           | ---          | ---        |
| CNN       |               |              | 
| ResNet-50 | 72.60         | 87.18        | 90.57      |
| ViT       | 90.11         | 96.06        | 97.16      |

> Note that ResNet and ViT are pretrained on ImageNet.

## Acknowledgements 

Much of the code on prototypical network is copied from THUNLP's [IsOBS](https://github.com/thunlp/IsOBS). 



