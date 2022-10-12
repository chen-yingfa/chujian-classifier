from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        output_dir: Path,
        num_epochs: int,
        batch_size: int,
        lr: float,
    ):
        self.model = model
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        
    def train(
        train_data: Dataset,
        dev_data: Dataset,
    ):
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        
    