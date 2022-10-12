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
        log_interval: int = 10,
        device: str = 'cuda',
    ):
        self.model = model
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.to(device)

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        self.cur_step = 0
        self.total_loss = 0
        print('------ Training ------')
        for batch in train_loader:
            self.cur_step += 1
            self.train_step(batch)
        print('------ Training Done ------')

    def train_step(self, batch):
        inputs, labels = batch
        inputs.to(self.device)
        labels.to(self.device)
        # Forward pass
        logits = self.model(inputs)
        loss = self.loss_fn(logits, labels)
        self.total_loss += loss.item()

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.cur_step += 1

        if self.cur_step % self.log_interval == 0:
            print({
                'epoch': {self.cur_ep},
                'step': self.cur_step,
                'loss': self.total_loss / self.cur_step,
            })

    def train(
        self,
        train_data: Dataset,
        dev_data: Dataset,
    ):
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        self.cur_ep = 0

        while self.cur_ep < self.num_epochs:
            self.train_epoch(train_loader)
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
        print('------ Evaluating ------')
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                logits = self.model(inputs)
                loss = self.loss_fn(logits, labels)
                all_labels += labels.tolist()
                topk_preds = torch.topk(logits, 10, dim=1)  # (B, k)
                all_preds += topk_preds.indices.tolist()
                
                total_loss += loss.item()
        print('------ Evaluation Done ------')
        return {
            'loss': total_loss / len(loader),
            'preds': all_preds,
        }
