from torchvision import transforms as T
import numpy as np
import torch
import json


def mean(arrs):
    return sum(arrs) * 1.0 / len(arrs)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


transform = T.Compose([
    T.Resize([50, 50]),
    # T.Grayscale(),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


def dump_list(path, lst):
    with open(path, 'w') as f:
        for x in lst:
            f.write(f"{x}\n")


def dump_json(data, file: str):
    json.dump(data, open(file, 'w'), indent=4)
