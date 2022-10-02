import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and
    'num_samples', in fact at every iteration the batch indexes will refer to
    'num_support' + 'num_query' samples for 'classes_per_it' random classes.
    '''
    def __init__(
        self,
        labels: list,
        classes_per_it: int,
        num_samples: int,
        num_iters: int,
    ):
        '''
        Initialize the PrototypicalBatchSampler object

        Params:
        - labels: an iterable containing label for each example in the dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class
        (support + query)
        - num_iters: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.num_iters = num_iters

        # 类，与类的数量
        self.classes, self.counts = np.unique(
            self.labels, return_counts=True)  # 每个标签有多少个数量
        self.classes = torch.LongTensor(self.classes)

        '''
        1. Create a matrix `indexes`, (# class, max # elements in class)
        2. Fill it with `np.nan`
        3. For every class c, fill the relative row with the indices samples
           belonging to c
        4. In numel_per_class we store the number of samples for each class/row
        '''
        # The i'th row stores the example indices belonging to class i.
        self.class_to_idxs = np.empty(
            shape=(len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.class_to_idxs = torch.Tensor(self.class_to_idxs)
        # The i'th element is the number of examples with class i.
        self.class_to_numels = torch.zeros_like(self.classes)

        # Loop all example labels (in the order of the dataset)
        for idx, label in enumerate(self.labels):
            # Get the class index of this label
            class_idx = np.argwhere(self.classes == label).item()
            # 把第一个是 nan 的替换成 idx
            # 第 class_idx 行就是属于这个 class 的所有样本的 index
            first_nan_col = np.where(
                np.isnan(self.class_to_idxs[class_idx]))[0][0]
            self.class_to_idxs[class_idx, first_nan_col] = idx
            # label_idx 总共有多少个属于它的数据
            self.class_to_numels[class_idx] += 1

    def __iter__(self) -> torch.LongTensor:
        '''
        yield a batch of indexes
        '''
        # support的数量 + query的数量（default: 5+5）
        samples_per_class = self.sample_per_class
        num_classes = self.classes_per_it  # (default: 50)

        for it in range(self.num_iters):
            batch_size = samples_per_class * num_classes
            # print('batch size:', batch_size)

            batch = torch.LongTensor(batch_size)
            # Random sample `num_classes` class indices.
            class_idxs = torch.randperm(len(self.classes))[:num_classes]

            # For each class, sample `samples_per_class` indices,
            # insert them into a slice of the batch.
            cur_slice_lo = 0
            # for i, c in enumerate(self.classes[class_indices]):
            for class_idx in class_idxs:
                # 抽取哪一行（哪个label）  译者注owo:找到classes中标号为c的元素下标
                # label_idx = torch.arange(len(self.classes)).long()[
                #     self.classes == c].item()

                # Why not just label_idx = c.item() ???
                '''
                If self.classes = [1, 2, 5]
                class_indices = [0, 2]
                then during i=0, c=1, during i=1, c=5
                so when i=0:
                    self.classes == c will return [True, False, False]
                    => label_idx = 0
                when i=1:
                    self.classes == c will return [False, False, True]
                    => label_idx = 2
                '''
                class_num_examples = self.class_to_numels[class_idx]

                # Sample `samples_per_class` examples for this class.
                col_idxs = torch.randperm(
                    class_num_examples)[:samples_per_class]
                # print('example_idxs', example_idxs)
                # print(self.labels[example_idxs])
                # print('batch.shape', batch.shape)
                # print('num_examples', num_examples)

                # Insert into batch
                slice_size = col_idxs.shape[0]
                cur_slice = slice(cur_slice_lo, cur_slice_lo + slice_size)
                batch[cur_slice] = self.class_to_idxs[class_idx][col_idxs]
                cur_slice_lo += slice_size
            # Shuffle batch
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.num_iters
