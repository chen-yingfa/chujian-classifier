import os
from pathlib import Path
from argparse import Namespace
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from modeling.model import ProtoNet, prototypical_loss, euclidean_dist
from arguments import get_parser
from sampler import PrototypicalBatchSampler
from utils import set_seed, mean
from dataset import ChujianDataset


def get_dataloader(
    args: Namespace,
    mode: str,
    img_size: tuple,
) -> DataLoader:
    '''Return dataloader'''
    assert Path(args.train_dir).exists()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # TODO: Add more transformations: data augmentation, normalize etc.
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomInvert(),
        transforms.RandomAutocontrast(),
        transforms.RandomGrayscale(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # dataset = datasets.ImageFolder(args.train_dir, transform=transform)
    dataset = ChujianDataset(args.train_dir, transform=transform)
    num_classes = len(dataset.classes)
    num_examples = len(dataset)
    print(f'# examples: {num_examples}')
    print(f"# classes: {mode} {num_classes}")
    if (num_classes < args.classes_per_it_tr or
            num_classes < args.classes_per_it_val):
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
        samples_per_class=num_samples,
        num_iters=args.iterations,
    )
    print('Instantiating DataLoader')

    def collate_fn(batch) -> Tuple[Tensor, Tensor]:
        inputs = torch.stack([x[0] for x in batch])
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
    device = 'cpu'
    if args.cuda and torch.cuda.is_available():
        device = 'cuda'
    loss_fn = prototypical_loss
    model.to(device)

    log_interval = 4
    steps_per_spoch = len(train_loader)

    print('------ Training ------')
    print(f'Log interval: {log_interval}')
    print(f'Steps per epoch: {steps_per_spoch}')

    for ep in range(args.epochs):
        print(f'------ Epoch: {ep} ------')
        acc_statistic = []
        loss_statistic = []

        model.train()
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs, labels)
            model_output = model(inputs)

            # Forward
            loss, acc = loss_fn(model_output, labels=labels,
                                num_support=args.num_support_tr)

            # Backward
            loss.backward()
            optim.step()

            # Log
            if step % log_interval == 0:
                log_stats = {
                    'epoch': round(ep + step / steps_per_spoch, 2),
                    'step': step,
                    'acc': round(acc.item(), 4),
                    'loss': round(loss.item(), 5),
                }
                print(log_stats)

            acc_statistic.append(acc.item())
            loss_statistic.append(loss.item())

        print('Average Train Loss: {}, Avg Train Accuracy: {}'
              .format(mean(loss_statistic), mean(acc_statistic)),
              end='\n\n')

        ckpt_file = Path(args.ckpt_dir, f'ckpt_{ep}.pt')
        torch.save(model.state_dict(), ckpt_file)
        scheduler.step()
    return


def initialize(
    args: Namespace
) -> Tuple[
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

    print('Getting model')
    model = ProtoNet()

    print('Preparing optimizer and scheduler')
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        gamma=args.lr_scheduler_gamma,
        step_size=args.lr_scheduler_step)
    return model, optim, scheduler


def get_device(cuda) -> str:
    if cuda and torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def get_representation(
    model: ProtoNet,
    dataset: ChujianDataset,
    device: str,
) -> Tensor:
    '''
    Get the prototype representation of the dataset. This will feed all
    examples in the dataset to the model, then average the output.

    Return:
        Tensor: shape (num_features)
    Args:
    '''
    data_loader = DataLoader(dataset, batch_size=50)
    model.eval()
    all_outputs = None
    for i, batch in enumerate(data_loader):
        inputs = batch.to(device)
        # print('inputs.size', inputs.size())
        outputs = model(inputs).detach().cpu()
        if i == 0:
            all_outputs = outputs
        else:
            all_outputs = torch.cat((all_outputs, outputs), dim=0)
    prototypes = all_outputs.to('cpu').mean(0)
    return prototypes


def get_all_prototypes(
    model: ProtoNet,
    dataset: ChujianDataset,
    num_classes: int,
    device: str,
    img_size: tuple,
    hidden_dim: int,
) -> Tensor:
    '''
    Get prototypes for each class. This will feed all examples in the dataset
    to the model, then average the output for each class.

    Returns:
        Tensor of shape (num_classes, hidden_dim)
    '''
    model.eval()
    prototypes = torch.empty(num_classes, hidden_dim)
    lo = 0
    for class_idx in range(num_classes):
        # Get a dataset of all examples belonging to this class
        hi = lo
        while hi < len(dataset) and dataset[hi][1] == class_idx:
            hi += 1
        class_size = hi - lo
        if class_size == 0:
            # This class has no examples in training data
            prototypes[class_idx] = torch.zeros(hidden_dim)
            # raise ValueError(f'Class {class_idx} has no examples')
            continue
        class_dataset = torch.empty(class_size, 3, *img_size)
        for i in range(lo, hi):
            class_dataset[i - lo] = dataset[i][0].unsqueeze(0)
        lo = hi
        # Get the protoype of this class
        prototypes[class_idx] = get_representation(
            model, class_dataset, device)
    return prototypes


def test(
    args: Namespace,
    model: ProtoNet,
    num_classes: int,
    img_size: tuple = (50, 50),
    hidden_dim: int = 576,
) -> None:
    '''
    Perform test on model
    '''
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    device = get_device(args.cuda)
    # Use training data to get prototypes for each class.
    # TODO: This needs to take into account for classes that are not in the
    # training set.
    train_dataset = ChujianDataset(args.train_dir, transform=transform)
    print('Getting prototypes from training data')
    prototypes = get_all_prototypes(
        model,
        dataset=train_dataset,
        num_classes=num_classes,
        device=device,
        img_size=img_size,
        hidden_dim=hidden_dim,
    )
    print("prototypes", prototypes.size())  # (p, d)

    batch_size = 50
    dataset = ChujianDataset(args.test_dir, transform=transform)
    num_examples = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size)

    print('------ Testing ------')
    print(f'Batch size: {batch_size}')
    print(f'Number of classes: {num_classes}')
    print(f'# examples {len(dataset)}')

    topks = [1, 3, 5, 10]
    num_corrects = {k: 0 for k in topks}

    def get_num_correct(
        labels: Tensor,
        dists: Tensor,
        k: int,
    ) -> int:
        '''
        Return the number of correct predictions
        '''
        log_p_y = F.log_softmax(-dists, dim=1)
        _, indices = torch.topk(log_p_y, k=k, dim=1)
        # Change (B) -> (B, 1), for comparison with log_p_y_k (B, k)
        labels = labels.view(-1, 1)
        return (labels == indices).sum().item()

    for batch in loader:
        inputs, labels = batch
        inputs = inputs.to(device)                      # (b, 3, h, w)
        outputs = model(inputs).detach().cpu()          # (b, d)
        dists = euclidean_dist(outputs, prototypes)     # (b, p)

        # top-k accuracy
        for k in topks:
            num_corrects[k] += get_num_correct(labels, dists, k)

    result = {
        'acc': {
            f'top-{k}': round(
                100 * num_corrects[k] / num_examples,
                2,
            ) for k in topks
        }
    }
    return result


def load_model(ckpt_file: Path) -> ProtoNet:
    model = ProtoNet()
    state_dict = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def main() -> None:
    args = get_parser().parse_args()
    print('------ args ------')
    for k, v in vars(args).items():
        print(f'{k:>20}: {v}')
    print('------------------')
    model, optim, scheduler = initialize(args)
    IMG_SIZE = (50, 50)

    if 'train' in args.mode:
        print('Getting train dataloader')
        train_loader = get_dataloader(args, 'train', IMG_SIZE)
        train(
            args=args,
            train_loader=train_loader,
            model=model,
            optim=optim,
            scheduler=scheduler,
        )

    if 'test' in args.mode:
        NUM_CLASSES = 525
        device = get_device(args.cuda)
        # Test on all epochs
        for ep in range(args.epochs):
            ckpt_file = Path(args.ckpt_dir, f'ckpt_{ep}.pt')
            model = load_model(ckpt_file)
            model.to(device)
            print(f"------ testing {ckpt_file} ------")
            result = test(
                args,
                model,
                num_classes=NUM_CLASSES,
                img_size=IMG_SIZE,
            )
            print(result)


if __name__ == '__main__':
    main()
