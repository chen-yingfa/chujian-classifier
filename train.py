# import torch.nn.functional as F
import torch
import os
from collections import Counter

import numpy as np
from tqdm import tqdm
# from collections import *

# from dataloader import *
# from testModel import test_model
from modeling.model import ProtoNet, prototypical_loss, euclidean_dist
# from source.parser import get_parser
# from source.sampler import PrototypicalBatchSampler
from arguments import get_parser
from sampler import PrototypicalBatchSampler
from utils import set_seed, mean, transform

# from torch.utils import data
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms as T




device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def init_dataloader(opt, mode):
    # dataset
    global n_classes
    dataset = datasets.ImageFolder(
        "./data_oct/" + mode + "_" + opt.oracle, transform=transform)
    n_classes = len(np.unique([dataset[i][1] for i in range(len(dataset))]))
    print("n_classes", mode, n_classes)  # 361
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    # sampler

    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:  # validation和test用同一组..？
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val
    labels = [dataset[i][1] for i in range(len(dataset))]
    # print ( classes_per_it, num_samples )  # 60  , 10
    # classes_per_it_tr number of random classes per episode for training, default=60'
    # num_samples = opt.num_support_tr + opt.num_query_tr
    # 一个batch 内的样本数也就是 classes_per_it_tr *  num_samples

    batch_sampler = PrototypicalBatchSampler(labels=labels,
                                             classes_per_it=classes_per_it,
                                             num_samples=num_samples,
                                             iterations=opt.iterations)
    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler)
    # for e in dataloader:
    #     #print (e )
    #     asf
    return dataloader


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    loss_fn = prototypical_loss

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        acc_statistic = []
        loss_statistic = []

        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch  # 应该可以用吧？
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()

            acc_statistic.append(acc.item())
            loss_statistic.append(loss.item())

        print('Average Train Loss: {}, Avg Train Accuracy: {}'
              .format(mean(loss_statistic), mean(acc_statistic)),
              end='\n\n')

        torch.save(
            model.state_dict(),
            os.path.join(opt.experiment_root, 'model_{}.pth'.format(epoch))
        )

        lr_scheduler.step()

    return 0


def initialize(options):
    '''
    初始化训练所需的模型及数据集
    '''
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    set_seed(options)
    tr_dataloader = init_dataloader(options, 'train')
    model = ProtoNet().to(device)
    optim = torch.optim.Adam(params=model.parameters(),
                             lr=options.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                   gamma=options.lr_scheduler_gamma,
                                                   step_size=options.lr_scheduler_step)

    return tr_dataloader, model, optim, lr_scheduler


def main():
    options = get_parser().parse_args()

    tr_dataloader, model, optim, lr_scheduler = initialize(options)
    for batch in tr_dataloader:
        x, y = batch
        print(x.shape, y.shape)
        print(Counter(y.numpy()))
        print(len(Counter(y.numpy())))
    # train(
    #     opt=options,
    #     tr_dataloader=tr_dataloader,
    #     val_dataloader=None,
    #     model=model,
    #     optim=optim,
    #     lr_scheduler=lr_scheduler
    # )

    # 加载训练好的model
    device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'
    model = ProtoNet().to(device)
    for epoch in range(options.epochs):
        print("===========model No.{}===========".format(epoch))
        # model.load_state_dict(
        #     torch.load("./model_save/model_{}.pth".format(epoch))
        # )
        # test_model(options, model,n_classes = n_classes)
        raise NotImplementedError


if __name__ == '__main__':
    main()
