import torch
from torch import Tensor, nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Consists of 4 conv blocks and 1 fully connected layer
    '''

    def __init__(
        self,
        # img_size: tuple = (96, 96),
        input_chan: int = 3,
        hidden_chan: int = 64,
        conv_output_chan: int = 64,
        # output_dim: int = 576,
    ):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(input_chan, hidden_chan),     # -> (B, 64, H/2, W/2)
            conv_block(hidden_chan, hidden_chan),   # -> (B, 64, H/4, W/4)
            conv_block(hidden_chan, hidden_chan),   # -> (B, 64, H/8, W/8)
            conv_block(hidden_chan, conv_output_chan),     # -> (B, 32, H/16, W/16)
        )
        # h = img_size[0]
        # w = img_size[1]
        # flat_dim = z_dim * h // 16 * w // 16
        # self.fc = nn.Linear(flat_dim, output_dim)

    def forward(self, x):
        # print(x.size())           # x: (B, 3, H, W)
        x = self.encoder(x)         # -> (B, 64, H/16, W/16)
        x = x.view(x.size(0), -1)   # -> (B, z_dim * H/16 * W/16)
        # x = self.fc(x)              # -> (B, output_dim)
        return x


class PrototypicalLoss(nn.Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined
    below
    '''

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
    '''
    Compute euclidean distance between two tensors

    Params:
    - x: Tensor of shape (n, d), n is the number of samples, d is the feature.
    - y: Tensor of shape (c, d), representation vectors of prototypes.
    '''
    n = x.size(0)  # 300
    c = y.size(0)  # 60
    d = x.size(1)  # 576
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, c, d)
    y = y.unsqueeze(0).expand(n, c, d)
    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(
    hidden: Tensor,
    labels: Tensor,
    num_support: int
):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned

    Params:
    - hidden: the model output for a batch of samples
    - labels: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''  # noqa
    labels = labels.to('cpu')   # (B)
    hidden = hidden.to('cpu')   # (B, C)

    # FIXME when torch.unique will be available on cuda too
    # 返回有哪些 class（returns the unique elements of the input tensor）
    classes = torch.unique(labels)
    n_classes = len(classes)

    # FIXME when torch will support where as np
    # assuming n_query, n_target constants

    # Occurrence of class[0] in labels - n_support
    n_query = labels.eq(classes[0].item()).sum().item() - num_support

    # 每一个 class 的 n_query + n_support 数目都是一样的，
    # 所以取第一个 class 来算就行了，
    # 在这个 episode 里面取到的总样本数减去 n_support 就是 n_query
    support_idxs = [
        labels.eq(c).nonzero()[:num_support].squeeze(1) for c in classes]
    # 对于每一个 class（label），
    # .eq 判断 target 的每一位是否是这个 label，
    # .nonzero: return indices of all nonzero elements
    # [:n_support] 从里面取出 n_support 作为支撑集，
    # squeeze 将其从 (n_support, 1) 压缩为 n_support 的 tensor

    prototypes = torch.stack([hidden[idx_list].mean(
        0) for idx_list in support_idxs])  # 计算每一个label的c_k（representation）

    # FIXME when torch will support where as np
    # query_idxs = torch.stack(
    #     list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))
    # )
    query_idxs = torch.stack(
        [labels.eq(c).nonzero()[num_support:] for c in classes])

    query_idxs = query_idxs.view(-1)

    # 获取查询集的对应编号，但是最后那个view(-1)啥意思啊

    query_samples = hidden.to('cpu')[query_idxs]
    # 是按class的顺序排的

    # (n_query*n_classes)*n_classes
    dists = euclidean_dist(query_samples, prototypes)

    # n_classes*n_query*n_classes
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)  # 从0~n_classes-1的tensor
    # n_classes*1*1，内容从0~n_classes-1
    target_inds = target_inds.view(n_classes, 1, 1)
    # n_claasses*n_query*1，从0~n_classes-1每个都有n_query个
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    # torch.max(Tensor,dim)返回两个值，第一个为储存着最大值的Tensor，第二维为储存着最大值对应的index的Tensor
    _, y_hat = log_p_y.max(2)
    # y_hat:n_classes*n_query
    # target_inds:n_classes*n_query*1
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val
