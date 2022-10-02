import os
from argparse import Namespace
from typing import Tuple

from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from modeling.model import ProtoNet, prototypical_loss
from arguments import get_parser
from sampler import PrototypicalBatchSampler
from utils import set_seed, mean


def get_dataloader(args: Namespace, mode: str) -> DataLoader:
    '''Return dataloader'''
    global n_classes
    dataset = datasets.ImageFolder(args.data_dir)
    n_classes = len(dataset.classes)
    n_examples = len(dataset)
    print(f'# examples: {n_examples}')
    print(f"# classes: {mode} {n_classes}")  # 361
    if (n_classes < args.classes_per_it_tr or
            n_classes < args.classes_per_it_val):
        raise Exception(
            'There are not enough classes in the dataset in order '
            'to satisfy the chosen classes_per_it. Decrease the '
            'classes_per_it_{tr/val} option and try again.'
        )

    # Sampler
    print('Building sampler')
    if 'train' in mode:
        classes_per_it = args.classes_per_it_tr
        num_samples = args.num_support_tr + args.num_query_tr
    else:  # validation 和 test 用同一组..？
        classes_per_it = args.classes_per_it_val
        num_samples = args.num_support_val + args.num_query_val
    labels = [dataset[i][1] for i in range(len(dataset))]
    # print(classes_per_it, num_samples)  # 60 10
    # classes_per_it_tr: number of random classes per episode for training
    # default=60
    # num_samples = opt.num_support_tr + opt.num_query_tr
    # 一个 batch 内的样本数也就是 classes_per_it_tr *  num_samples

    print('Instantiating PrototypicalBatchSampler')
    batch_sampler = PrototypicalBatchSampler(
        labels=labels,
        classes_per_it=classes_per_it,
        num_samples=num_samples,
        num_iters=args.iterations,
    )
    print('Instantiating DataLoader')
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.PILToTensor(),
        # TODO: Add more transformations: data augmentation, normalize etc.
    ])

    def collate_fn(batch) -> Tuple[Tensor, Tensor]:
        inputs = torch.stack([transform(x[0]) for x in batch])
        labels = torch.LongTensor([x[1] for x in batch])
        return inputs, labels

    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
    )
    return dataloader


def train(
    args: Namespace,
    train_loader: DataLoader,
    model: nn.Module,
    optim: torch.optim.Adam,
    scheduler: torch.optim.lr_scheduler.StepLR,
    val_loader=None,
) -> None:
    '''
    Train the model with the prototypical learning algorithm
    '''
    # device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    device = 'cuda'
    loss_fn = prototypical_loss

    log_interval = 2
    steps_per_spoch = len(train_loader)

    for ep in range(args.epochs):
        print(f'*** Epoch: {ep} ***')
        acc_statistic = []
        loss_statistic = []

        # train_iter = iter(train_loader)
        model.train()
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            x, y = batch  # 应该可以用吧？
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            
            # Forward
            loss, acc = loss_fn(model_output, labels=y,
                                num_support=args.num_support_tr)
            
            # Backward
            loss.backward()
            optim.step()

            # Log
            if step % log_interval == 0:
                log_stats = {
                    'epoch': ep + step / steps_per_spoch,
                    'step': step,
                    'acc': acc.item(),
                    'loss': loss.item(),
                }
                print(log_stats)

            acc_statistic.append(acc.item())
            loss_statistic.append(loss.item())

        print('Average Train Loss: {}, Avg Train Accuracy: {}'
              .format(mean(loss_statistic), mean(acc_statistic)),
              end='\n\n')

        torch.save(
            model.state_dict(),
            os.path.join(args.ckpt_dir, 'model_{}.pth'.format(ep))
        )
        scheduler.step()
    return


def initialize(
    args: Namespace
) -> Tuple[
    DataLoader,
    nn.Module,
    torch.optim.Adam,
    torch.optim.lr_scheduler.StepLR,
]:
    '''
    初始化训练所需的模型及数据集
    '''
    print('Initializing')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    print(f'Setting seed: {args.seed}', flush=True)
    set_seed(args.seed)

    print('Getting train dataloader')
    train_dataloader = get_dataloader(args, 'train')

    print('Getting model')
    model = ProtoNet()

    print('Preparing optimizer and scheduler')
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        gamma=args.lr_scheduler_gamma,
        step_size=args.lr_scheduler_step)
    return train_dataloader, model, optim, scheduler


def main():
    args = get_parser().parse_args()
    train_loader, model, optim, scheduler = initialize(args)
    print('Logging the shape of data')
    # for batch in train_loader:
    #     x, y = batch
    #     print('x.shape', x.shape)
    #     print('y.shape', y.shape)
    #     exit()

    # NOTE: no dev data loader
    train(
        args=args,
        train_loader=train_loader,
        model=model,
        optim=optim,
        scheduler=scheduler,
    )

    # TODO: Load best checkpoint
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    model = ProtoNet().to(device)
    for epoch in range(args.epochs):
        print("===========model No.{}===========".format(epoch))
        # model.load_state_dict(
        #     torch.load("./model_save/model_{}.pth".format(epoch))
        # )
        # test_model(options, model,n_classes = n_classes)
        raise NotImplementedError


if __name__ == '__main__':
    main()
