from pathlib import Path
import time
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        output_dir: Path,
        num_epochs: int = 2,
        batch_size: int = 4,
        lr: float = 0.0001,
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
        
        output_dir.mkdir(exist_ok=True, parents=True)
        self.train_log_path = output_dir / 'train.log'
        
        # Dump training args
        train_args = {
            k: str(vars(self)[k]) for k in [
                'batch_size', 
                'num_epochs', 
                'output_dir', 
                'lr', 
                'log_interval',
                'device',
            ]
        }
        args_file = output_dir / 'train_args.json'
        json.dump(train_args, args_file.open('w'), indent=4)
    
    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.train_log_file)

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        self.cur_step = 0
        self.total_loss = 0
        self.train_start_time = time.time()
        self.log(f'Start epoch {self.cur_ep}')
        for batch in train_loader:
            self.cur_step += 1
            self.train_step(batch)
        self.log(f'Epoch done')

    def train_step(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
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
            self.log({
                'epoch': self.cur_ep,
                'step': self.cur_step,
                'loss': round(self.total_loss / self.cur_step, 4),
                'time': round(time.time() - self.train_start_time, 2),
            }, flush=True)

    def train(
        self,
        train_data: Dataset,
        dev_data: Dataset,
    ):
        self.train_log_file = open(self.train_log_path, 'w')
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True,
        )
        self.cur_ep = 0
        
        self.log('------ Training ------')
        self.log(f'  Num steps: {len(train_loader)}')
        self.log(f'  Num examples: {len(train_data)}')
        self.log(f'  Num epochs: {self.num_epochs}')
        self.log(f'  Batch size: {self.batch_size}')
        self.log(f'  Log interval: {self.log_interval}')

        while self.cur_ep < self.num_epochs:
            self.train_epoch(train_loader)
            self.validate(dev_data)
            self.cur_ep += 1
        self.log('------ Training Done ------')
        self.train_log_file.close()

    def validate(self, dev_data: Dataset):
        dev_dir = self.output_dir / f'ckpt_{self.cur_ep}'
        result = self.evaluate(dev_data, dev_dir)
        del result['preds']
        result_file = dev_dir / 'result.json'
        json.dump(result, open(result_file, 'w'), indent=4)

    def save_ckpt(self, filename: str):
        ckpt_dir = self.output_dir / 'ckpts'
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), ckpt_dir / filename)

    def load_ckpt(self, filename: str) -> nn.Module:
        sd = torch.load(self.output_dir / filename)
        self.model.load_state_dict(sd)

    def evaluate(
        self,
        dataset: Dataset, 
        test_dir: Path,
    ):
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.model.eval()
        print('------ Evaluating ------')
        print(f'Num steps: {len(loader)}')
        print(f'Num examples: {len(dataset)}')
        
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                logits = self.model(inputs.to(self.device))
                loss = self.loss_fn(logits, labels.to(self.device))
                all_labels += labels.tolist()
                topk_preds = torch.topk(logits, 10, dim=1)  # (B, k)
                all_preds += topk_preds.indices.tolist()
                
                total_loss += loss.item()
                
        preds_file = test_dir / 'preds.json'
        json.dump(all_preds, open(preds_file, 'w'), indent=4)
        # Compute top-k accuracy
        acc = {}
        for k in [1, 3, 5, 10]:
            acc[k] = 0
            for label, preds in zip(all_labels, all_preds):
                if label in preds[:k]:
                    acc[k] += 1
            acc[k] /= len(all_labels)
        print('------ Evaluation Done ------')
        
        
        return {
            'loss': total_loss / len(loader),
            'preds': all_preds,
            'acc': acc,
        }
