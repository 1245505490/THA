import os

import torch
import numpy as np
from preprocess import load_sparse
import _pickle as pickle
import torch.nn.functional as F


class EHRDataset:
    def __init__(self, data_path, label='m', batch_size=32, shuffle=True, device=torch.device('cpu')):
        super().__init__()
        self.path = data_path
        self.is_train = True if 'train' in self.path else False
        self.code_x, self.visit_lens, self.intervals, self.disease_x, self.disease_lens, self.drug_x, self.drug_lens, self.mark, self.y, self.tf_idf_weight = self._load(
            label)
        self._size = self.code_x.shape[0]
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def _load(self, label):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = np.load(os.path.join(self.path, 'visit_lens.npz'))['lens']
        intervals = load_sparse(os.path.join(self.path, 'intervals.npz'))
        if label == 'm':
            y = load_sparse(os.path.join(self.path, 'y.npz'))
        elif label == 'h':
            y = np.load(os.path.join(self.path, 'hf_y.npz'))['hf_y']
        else:
            raise KeyError('Unsupported label type')
        disease_dataset = pickle.load(open(os.path.join(self.path, 'disease_dataset.pkl'), 'rb'))
        disease_x, disease_lens = disease_dataset
        drug_dataset = pickle.load(open(os.path.join(self.path, 'drug_dataset.pkl'), 'rb'))
        tf_idf_weight = None
        if self.is_train:
            drug_x, drug_lens, tf_idf_weight = drug_dataset
        else:
            drug_x, drug_lens = drug_dataset

        mark = load_sparse(os.path.join(self.path, 'mark.npz'))
        return code_x, visit_lens, intervals, disease_x, disease_lens, drug_x, drug_lens, mark, y, tf_idf_weight

    def on_epoch_end(self):
        # 每个epoch结束执行该函数
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        return self.y

    def __len__(self):
        # 返回批数量
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        intervals = torch.from_numpy(self.intervals[slices]).to(device=device, dtype=torch.float32)
        disease_x = torch.from_numpy(self.disease_x[slices]).to(device=device, dtype=torch.int32)
        disease_lens = torch.from_numpy(self.disease_lens[slices]).to(device=device, dtype=torch.int32)
        drug_x = torch.from_numpy(self.drug_x[slices]).to(device=device, dtype=torch.int32)
        drug_lens = torch.from_numpy(self.drug_lens[slices]).to(device=device, dtype=torch.int32)
        mark = torch.from_numpy(self.mark[slices]).to(device=device, dtype=torch.float32)
        if self.is_train:
            tf_idf_weight = torch.from_numpy(self.tf_idf_weight[slices]).to(device=device,dtype=torch.float32)
            return code_x, visit_lens, intervals, disease_x, disease_lens, drug_x, drug_lens, mark, y, tf_idf_weight
        else:
            return code_x, visit_lens, intervals, disease_x, disease_lens, drug_x, drug_lens, mark, y


def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str


def medical_codes_loss(y_pred, y_true):
    # 计算每个样本的损失
    per_sample_losses = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    # 对每个样本的损失求和
    summed_losses = per_sample_losses.sum(dim=-1)
    # 对所有样本的损失求平均
    mean_loss = summed_losses.mean()
    return mean_loss


class MultiStepLRScheduler:
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_lr = init_lr
        self.lrs = self._generate_lr(milestones, lrs)
        self.current_epoch = 0

    def _generate_lr(self, milestones, lrs):
        # 将 milestones 列表前后加入 1 和 epochs + 1，形成完整的 epoch 划分区间
        # [1,20,30,201]
        milestones = [1] + milestones + [self.epochs + 1]
        # 将初始学习率加入 lrs 列表开头，形成完整的对应关系
        # [0.01,0.0001,0.000001]
        lrs = [self.init_lr] + lrs

        # 于是会生成一个np列表，有20个0.01 10个0.0001,171个0.000001 对应了每一轮的学习率
        lr_grouped = np.concatenate([np.ones((milestones[i + 1] - milestones[i],)) * lrs[i]
                                     for i in range(len(milestones) - 1)])
        return lr_grouped

    def step(self):
        lr = self.lrs[self.current_epoch]
        for group in self.optimizer.param_groups:
            # 设置该轮的学习率
            group['lr'] = lr
        self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0


def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str
