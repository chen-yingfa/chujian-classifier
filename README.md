# Chujian Classifier

Implementation of Prototypical Network on classification of Chujian (楚简) characters. Support Prototypical Network (based on CNN), ResNet and ViT.

## Data

Preprocess the data into specific format with the code in `data` folder. See `data/README.md` for details.

The data should to organized into JSON files containing a dictionary where each key is a glyph, and the value is a list of image files of that glyph. For example:

```json
{
    "{阤土}": [
        "/path/to/glyphs/{阤土}/清華簡_04A-筮法_58_04A-58-09.png"
    ],
    "囡": [
        "/path/to/glyphs/囡/清華簡_03F-良臣_2_03F-02-01.png"
    ],
    "緅": [
        "/path/to/glyphs/○（緅）/曾侯乙簡_1-A組_07_1-007-09.png",
        "/path/to/glyphs/緅/包山簡_3-遣冊賵書_D_3-D-150.png"
    ],
    // ...
}
```

There should be three JSON files, corresponding to training, validation and testing sets.

A small part (1,000 examples) of the entire Chujian dataset is extracted for convenience, it is located in `data/chujian/glyphs_small`. 

## Usage

### Training

Set the data path with `--train_json` and `--test_json`.
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
| $k=3$
| ResNet-50 |    |
| ViT       |    |
| $k=10$
| ResNet-50 | 72.60         | 87.18        | 90.57      |
| ViT       | 90.11         | 96.06        | 97.16      |

> Note that ResNet and ViT are pretrained on ImageNet.

## Acknowledgements 

Much of the code on prototypical network is copied from THUNLP's [IsOBS](https://github.com/thunlp/IsOBS). 



