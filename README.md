# Chujian Classifier

Implementation of Prototypical Network on classification of Chujian (楚简) characters.

## Data

A small part (1,000 examples) of the entire Chujian dataset is extracted for convenience, it is located in `data/chujian/glyphs_small`. 

## Usage

### Training

Set the data path with `--train_dir` and `--test_dir`.
Set the output path with `--output_dir`.

Execute

```bash
python3 train.py
```

### Testing

```bash
python3 train.py --mode test
```

## Acknowledgements 

Much of the code is copied from THUNLP's [IsOBS](https://github.com/thunlp/IsOBS). 



