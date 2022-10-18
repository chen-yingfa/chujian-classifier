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
        device: str = "cuda",
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
            self.optimizer,
            step_size=1,
            gamma=0.5,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.to(device)

        output_dir.mkdir(exist_ok=True, parents=True)
        self.train_log_path = output_dir / "train.log"

        # Dump training args
        train_args = {
            k: str(vars(self)[k])
            for k in [
                "batch_size",
                "num_epochs",
                "output_dir",
                "lr",
                "log_interval",
                "device",
            ]
        }
        args_file = output_dir / "train_args.json"
        json.dump(train_args, args_file.open("w"), indent=4)

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.train_log_file)

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        self.cur_step = 0
        self.total_loss = 0
        self.epoch_start_time = time.time()
        self.log(f"Start epoch {self.cur_ep}")
        for batch in train_loader:
            self.train_step(batch)
        self.scheduler.step()
        self.log("Epoch done")

    def train_step(self, batch: tuple):
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
        self.optimizer.zero_grad()

        self.cur_step += 1

        if self.cur_step % self.log_interval == 0:
            self.log(
                {
                    "epoch": round(
                        self.cur_ep + self.cur_step / len(self.train_loader), 3
                    ),
                    "step": self.cur_step,
                    "lr": round(self.scheduler.get_last_lr()[0], 6),
                    "loss": round(self.total_loss / self.cur_step, 4),
                    "time": round(time.time() - self.train_start_time),
                    "epoch_time": round(time.time() - self.epoch_start_time),
                },
                flush=True,
            )

    def train(
        self,
        train_data: Dataset,
        dev_data: Dataset,
    ):
        self.train_log_file = open(self.train_log_path, "w")
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.cur_ep = 0
        self.train_start_time = time.time()

        self.log("------ Training ------")
        self.log(f"  Num steps: {len(self.train_loader)}")
        self.log(f"  Num examples: {len(train_data)}")
        self.log(f"  Num epochs: {self.num_epochs}")
        self.log(f"  Batch size: {self.batch_size}")
        self.log(f"  Log interval: {self.log_interval}")

        while self.cur_ep < self.num_epochs:
            self.train_epoch(self.train_loader)
            self.validate(dev_data)
            self.cur_ep += 1
        self.log("------ Training Done ------")
        self.train_log_file.close()

    def validate(self, dev_data: Dataset):
        dev_dir = self.output_dir / f"ckpt_{self.cur_ep}"
        dev_dir.mkdir(exist_ok=True, parents=True)
        result = self.evaluate(dev_data, dev_dir)
        del result["preds"]
        result_file = dev_dir / "result.json"
        json.dump(result, open(result_file, "w"), indent=4)
        self.save_ckpt(dev_dir / "ckpt.pt")

    def save_ckpt(self, ckpt_file: Path):
        print(f"Saving checkpoint to {ckpt_file}")
        ckpt_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), ckpt_file)

    def load_ckpt(self, path: str) -> nn.Module:
        print(f"Loading checkpoint from {path}")
        sd = torch.load(self.output_dir / path)
        self.model.load_state_dict(sd)

    def evaluate(
        self,
        dataset: Dataset,
        output_dir: Path,
    ):
        eval_batch_size = 4 * self.batch_size
        loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        self.model.eval()
        self.log("------ Evaluating ------")
        self.log(f"Num steps: {len(loader)}")
        self.log(f"Num examples: {len(dataset)}")
        self.log(f"batch_size: {eval_batch_size}")

        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                inputs, labels = batch
                logits = self.model(inputs.to(self.device))
                loss = self.loss_fn(logits, labels.to(self.device))
                all_labels += labels.tolist()
                topk_preds = torch.topk(logits, 10, dim=1)  # (B, k)
                all_preds += topk_preds.indices.tolist()

                total_loss += loss.item()

                if (step + 1) % self.log_interval == 0:
                    self.log(
                        {
                            "step": step,
                            "loss": total_loss / (step + 1),
                        }
                    )

        preds_file = output_dir / "preds.json"
        json.dump(all_preds, open(preds_file, "w"), indent=4)
        # Compute top-k accuracy
        acc = {}
        for k in [1, 3, 5, 10]:
            acc[k] = 0
            for label, preds in zip(all_labels, all_preds):
                if label in preds[:k]:
                    acc[k] += 1
            acc[k] /= len(all_labels)
        self.log(acc)
        self.log("loss", total_loss / len(loader))
        self.log("------ Evaluation Done ------")

        return {
            "loss": total_loss / len(loader),
            "preds": all_preds,
            "acc": acc,
        }
