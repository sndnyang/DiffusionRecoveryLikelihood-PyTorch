import os
import torch
import torch as t
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np


def sqrt(x):
    return int(t.sqrt(t.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def get_data(args):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.ToTensor(),
             tr.Normalize(mean, std),
             # lambda x: x + args.sigma * t.randn_like(x)
             ]
        )
        transform_px = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize(mean, std),
             ]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize(mean, std),
             # lambda x: x + args.sigma * t.randn_like(x)
             ]
        )
        transform_px = tr.Compose(
            [tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize(mean, std),
             ]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize(mean, std),
         ]
    )

    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            args.n_classes = 10
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            args.n_classes = 100
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            args.n_classes = 10
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True, split="train" if train else "test")

    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(args.seed)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid > args.n_classes:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = train_inds

    dset_train = DataSubset(dataset_fn(True, transform_px), inds=train_inds)
    dset_train_labeled = DataSubset(dataset_fn(True, transform_train), inds=train_labeled_inds)
    dset_valid = DataSubset(dataset_fn(True, transform_test), inds=valid_inds)

    num_workers = 0 if args.debug else 4
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    label_bs = 128
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=label_bs, shuffle=True, num_workers=num_workers, drop_last=True)
    dload_train = cycle(dload_train)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid, dload_test


class EMA:
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
