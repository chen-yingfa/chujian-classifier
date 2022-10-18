import argparse


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="./data/chujian/glyphs_955/train")
    p.add_argument("--test_dir", default="./data/chujian/glyphs_955/test")
    p.add_argument("--output_dir", default="./result/glyphs_955")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", type=bool, default=True)
    p.add_argument("--mode", default="train_test")
    p.add_argument("--batch_size", type=int, default=512)
    # 改了，原100
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.0005)
    # 学习率衰减用的
    p.add_argument(
        "--lr_step",
        type=int,
        default=1,
        help="StepLR learning rate scheduler step, default=20",
    )

    p.add_argument(
        "--lr_gamma",
        type=float,
        default=0.8,
        help="StepLR learning rate scheduler gamma, default=0.5",
    )
    # 改了，原100
    p.add_argument(
        "-its",
        "--iterations",
        type=int,
        default=200,
        help="number of episodes per epoch, default=100",
    )

    p.add_argument(
        "-cTr",
        "--classes_per_it_tr",
        type=int,
        default=100,
        help="number of random classes per episode for training, default=60",
    )

    p.add_argument(
        "-nsTr",
        "--num_support_tr",
        type=int,
        default=5,
        help="number of samples per class to use as support for training",
    )
    p.add_argument(
        "-nqTr",
        "--num_query_tr",
        type=int,
        default=5,
        help="number of samples per class to use as query for training",
    )

    # No validation now...

    # p.add_argument(
    #     '-cVa', '--classes_per_it_val',
    #     type=int,
    #     help='number of classes per episode for validation, default=5',
    #     default=5,
    # )
    # # 改了，原5
    # p.add_argument(
    #     '-nsVa', '--num_support_val',
    #     type=int,
    #     help='number of samples per class to use as support for validation',
    #     default=4,
    # )
    # # 改了，原15
    # p.add_argument(
    #     '-nqVa', '--num_query_val',
    #     type=int,
    #     help='number of samples per class to use as query for validation',
    #     default=4,
    # )

    return p
