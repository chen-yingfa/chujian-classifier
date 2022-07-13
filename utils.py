from torchvision import transforms as T
import numpy as np
import torch


def mean(arrs):
    return sum(arrs) * 1.0 / len(arrs)


def set_seed(opt):
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


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