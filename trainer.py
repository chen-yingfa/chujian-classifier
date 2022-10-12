from pathlib import Path

import torch
from torch import nn
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

    def _train_epoch(self, train_loader: DataLoader):
        self.model.train()
        for batch in train_loader:
            self._train_batch(batch)

    def _train_batch(self, batch):
        image, label = batch
        outputs = self.model(image)
        print(outputs)
        exit()

    def train(
        self,
        train_data: Dataset,
        dev_data: Dataset,
    ):
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.cur_ep = 0

        while self.cur_ep < self.num_epochs:
            self._train_epoch(train_loader)
            self.cur_ep += 1
            self.save_ckpt(f'ckpt_{self.cur_ep}.pt')

    def save_ckpt(self, filename: str):
        torch.save(self.model.state_dict(), self.output_dir / filename)

    def load_ckpt(self, filename: str) -> nn.Module:
        sd = torch.load(self.output_dir / filename)
        self.model.load_state_dict(sd)

    def evaluate(self, dataset):
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                self.eval_step(batch)

    def eval_step(self, batch):
        image, label = batch
        outputs = self.model(image)
        print(outputs)
        exit()
