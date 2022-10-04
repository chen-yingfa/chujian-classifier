import os
from pathlib import Path
from argparse import Namespace
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from modeling.model import ProtoNet, prototypical_loss, euclidean_dist
from arguments import get_parser
from sampler import PrototypicalBatchSampler
from utils import set_seed, mean
from dataset import ChujianDataset


def get_dataloader(args: Namespace, mode: str) -> DataLoader:
    '''Return dataloader'''
    assert Path(args.data_dir).exists()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        # TODO: Add more transformations: data augmentation, normalize etc.
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomInvert(),
        transforms.RandomAutocontrast(),
        transforms.RandomGrayscale(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    # dataset = ChujianDataset(args.data_dir, transform=transform)
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

        ckpt_file = Path(args.ckpt_dir, f'ckpt_{ep}.pt')
        torch.save(model.state_dict(), ckpt_file)
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


def get_device(cuda) -> str:
    if cuda and torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def get_representation(
    args: Namespace,
    test_dataloader,
    model: ProtoNet,
    full_size: int,
) -> list:
    '''
    Return a vector
    '''
    model.eval()
    device = get_device(args.cuda)
    test_iter_c = iter(test_dataloader)
    print(full_size)
    model_output = torch.empty(min(full_size, 100), 576)

    first = True
    for batch_c in test_iter_c:
        x_c = batch_c
        print(x_c.size())
        x_c = x_c.to(device)
        if first:
            model_output = model(x_c).detach().to('cpu')
            first = False
        else:
            model_output = torch.cat(
                (model_output, model(x_c).detach().to('cpu')), dim=0)
    # print("model_output",model_output.size())
    support_cpu = model_output.to('cpu')
    prototypes = support_cpu.mean(0)
    # print("prototpyes",prototypes.size())
    return prototypes


def test(
    args: Namespace,
    model: ProtoNet,
    num_classes: int,
) -> None:
    '''
    Perform test on model
    '''

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model.eval()
    # dataset = datasets.ImageFolder(args.test_dir, transform=transforms)
    dataset = ChujianDataset(args.test_dir, transform=transform)
    prototypes = torch.empty(num_classes, 576)
    print(f'Prototypes size: {prototypes.size()}')

    # Get representation of prototypes
    head = 0
    tail = 0
    class_idx = 0
    # loop all classes
    while class_idx < num_classes:
        while tail < len(dataset) and dataset[tail][1] == class_idx:
            tail += 1
        class_dataset = torch.empty(tail - head, 3, 50, 50)
        for i in range(head, tail):
            class_dataset[i-head] = dataset[i][0].unsqueeze(0)
        prototypes[class_idx] = get_representation(
            args,
            DataLoader(class_dataset, batch_size=600),
            model,
            len(class_dataset),
        )
        head = tail
        class_idx += 1

    print("prototypes", prototypes.size())
    device = get_device(args.cuda)
    dataset = ChujianDataset(args.test_dir, transform=transform)
    num_correct_1 = 0
    num_correct_3 = 0
    num_correct_5 = 0
    num_correct_10 = 0
    num_examples = len(dataset)
    loader = DataLoader(dataset, batch_size=50)

    def get_num_correct(
        labels: Tensor,
        dists: Tensor,
        k: int,
    ) -> int:
        '''
        Return the number of correct predictions
        '''
        _, log_p_y_k = torch.topk(F.log_softmax(-dists, dim=1), k=k, dim=1)
        # Change (B) -> (B, 1), for comparison with log_p_y_k (B, k)
        labels = labels.view(-1, 1)
        return (labels == log_p_y_k).sum().item()

    for batch in loader:
        inputs, labels = batch  # x:数据，y:label
        inputs = inputs.to(device)
        print("inputs", inputs.size())

        outputs = model(inputs).detach().to('cpu')
        dists = euclidean_dist(outputs, prototypes)

        print('model_output shape:', outputs.shape)
        print('prototypes shape:', prototypes.shape)
        print("dists:", dists.size())  # dists torch.Size([600, 361])

        # top-1 accuracy
        log_p_y = F.log_softmax(-dists, dim=1).max(1)
        # print(answer_cpu.size(),log_p_y[1].size())
        # [1]表示取得是index，因为第0维是value
        num_correct_1 += labels.eq(log_p_y[1]).sum().item()

        if True:
            num_correct_1_0 = get_num_correct(labels, dists, 1)
            assert num_correct_1_0 == num_correct_1

        # top-k accuracy
        num_correct_3 += get_num_correct(labels, dists, 3)
        num_correct_5 += get_num_correct(labels, dists, 5)
        num_correct_10 += get_num_correct(labels, dists, 10)

    def get_result(topk: int, num_correct: int) -> dict:
        return {
            'topk': topk,
            'accuracy': round(100 * num_correct / num_examples, 2),
        }

    print(get_result(1, num_correct_1))
    print(get_result(3, num_correct_3))
    print(get_result(5, num_correct_5))
    print(get_result(10, num_correct_10))


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
    train_loader, model, optim, scheduler = initialize(args)

    # print('Logging the shape of data')
    # for batch in train_loader:
    #     x, y = batch
    #     print('x.shape', x.shape)
    #     print('y.shape', y.shape)
    #     exit()

    if 'train' in args.mode:
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
            test(args, model, num_classes=NUM_CLASSES)
            raise NotImplementedError


if __name__ == '__main__':
    main()
