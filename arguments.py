import argparse


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', default='./data/chujian/glyphs_small_train')
    p.add_argument('--test_dir', default='./data/chujian/glyphs_small_test')
    p.add_argument(
        '-exp', '--ckpt_dir',
        help='root where to store models, losses and accuracies',
        default='./ckpts')
    # 改了，原100
    p.add_argument(
        '--epochs', type=int, default=2,
        help='number of epochs to train for')
    p.add_argument(
        '--lr', type=float, default=0.001,
        help='learning rate for the model, default=0.001')
    # 学习率衰减用的
    p.add_argument(
        '-lrS', '--lr_scheduler_step', type=int, default=20,
        help='StepLR learning rate scheduler step, default=20')

    p.add_argument(
        '-lrG', '--lr_scheduler_gamma', type=float, default=0.5,
        help='StepLR learning rate scheduler gamma, default=0.5')
    # 改了，原100
    p.add_argument(
        '-its', '--iterations',
        type=int, default=100,
        help='number of episodes per epoch, default=100',
    )

    p.add_argument(
        '-cTr', '--classes_per_it_tr',
        type=int,
        default=20,
        help='number of random classes per episode for training, default=60',
    )

    p.add_argument(
        '-nsTr', '--num_support_tr',
        type=int,
        default=5,
        help='number of samples per class to use as support for training',
    )

    p.add_argument(
        '-nqTr', '--num_query_tr',
        type=int,
        default=5,
        help='number of samples per class to use as query for training',
    )

    p.add_argument(
        '-cVa', '--classes_per_it_val',
        type=int,
        help='number of random classes per episode for validation, default=5',
        default=5,
    )
    # 改了，原5
    p.add_argument(
        '-nsVa', '--num_support_val',
        type=int,
        help='number of samples per class to use as support for validation',
        default=4,
    )
    # 改了，原15
    p.add_argument(
        '-nqVa', '--num_query_val',
        type=int,
        help='number of samples per class to use as query for validation',
        default=4,
    )
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda', type=bool, default=True)
    p.add_argument('--mode', type=str, default='train_test')

    return p
