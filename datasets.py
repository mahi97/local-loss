import torch
import numpy as np
from torchvision import datasets, transforms


class Cutout(object):
    '''Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    '''

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        '''
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        '''
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class NClassRandomSampler(torch.utils.data.sampler.Sampler):
    r'''Samples elements such that most batches have N classes per batch.
    Elements are shuffled before each epoch.

    Arguments:
        targets: target class for each example in the dataset
        n_classes_per_batch: the number of classes we want to have per batch
    '''

    def __init__(self, targets, n_classes_per_batch, batch_size):
        self.targets = targets
        self.n_classes = int(np.max(targets))
        self.n_classes_per_batch = n_classes_per_batch
        self.batch_size = batch_size

    def __iter__(self):
        n = self.n_classes_per_batch

        ts = list(self.targets)
        ts_i = list(range(len(self.targets)))

        np.random.shuffle(ts_i)
        # algorithm outline:
        # 1) put n examples in batch
        # 2) fill rest of batch with examples whose class is already in the batch
        while len(ts_i) > 0:
            idxs, ts_i = ts_i[:n], ts_i[n:]  # pop n off the list

            t_slice_set = set([ts[i] for i in idxs])

            # fill up idxs until we have n different classes in it. this should be quick.
            k = 0
            while len(t_slice_set) < 10 and k < n * 10 and k < len(ts_i):
                if ts[ts_i[k]] not in t_slice_set:
                    idxs.append(ts_i.pop(k))
                    t_slice_set = set([ts[i] for i in idxs])
                else:
                    k += 1

            # fill up idxs with indexes whose classes are in t_slice_set.
            j = 0
            while j < len(ts_i) and len(idxs) < self.batch_size:
                if ts[ts_i[j]] in t_slice_set:
                    idxs.append(ts_i.pop(j))  # pop is O(n), can we do better?
                else:
                    j += 1

            if len(idxs) < self.batch_size:
                needed = self.batch_size - len(idxs)
                idxs += ts_i[:needed]
                ts_i = ts_i[needed:]

            for i in idxs:
                yield i

    def __len__(self):
        return len(self.targets)


class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]


def get_dataset(args):
    input_dim = None
    input_ch = None
    num_classes = None
    train_loader = None
    test_loader = None
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'MNIST':
        input_dim = 28
        input_ch = 1
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = datasets.MNIST('../data/MNIST', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels.numpy(),
                                                                                 args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data/MNIST', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'FashionMNIST':
        input_dim = 28
        input_ch = 1
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = datasets.FashionMNIST('../data/FashionMNIST', train=True, download=True,
                                              transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels.numpy(),
                                                                                 args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data/FashionMNIST', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.286,), (0.353,))
                                  ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'KuzushijiMNIST':
        input_dim = 28
        input_ch = 1
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1904,), (0.3475,))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = KuzushijiMNIST('../data/KuzushijiMNIST', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels.numpy(),
                                                                                 args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            KuzushijiMNIST('../data/KuzushijiMNIST', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1904,), (0.3475,))
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'CIFAR10':
        input_dim = 32
        input_ch = 3
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = datasets.CIFAR10('../data/CIFAR10', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels,
                                                                                 args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data/CIFAR10', train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                             ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'CIFAR100':
        input_dim = 32
        input_ch = 3
        num_classes = 100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = datasets.CIFAR100('../data/CIFAR100', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels,
                                                                                 args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../data/CIFAR100', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
                              ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'SVHN':
        input_dim = 32
        input_ch = 3
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = torch.utils.data.ConcatDataset((
            datasets.SVHN('../data/SVHN', split='train', download=True, transform=train_transform),
            datasets.SVHN('../data/SVHN', split='extra', download=True, transform=train_transform)))
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.labels,
                                                                                 args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN('../data/SVHN', split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
                          ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'STL10':
        input_dim = 96
        input_ch = 3
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = datasets.STL10('../data/STL10', split='train', download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.labels,
                                                                                 args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.STL10('../data/STL10', split='test',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
                           ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'ImageNet':
        input_dim = 224
        input_ch = 3
        num_classes = 1000
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
        dataset_train = datasets.ImageFolder('../data/ImageNet/train', transform=train_transform)
        labels = np.array([a[1] for a in dataset_train.samples])
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=None if args.classes_per_batch == 0 else NClassRandomSampler(labels, args.classes_per_batch,
                                                                                 args.batch_size),
            batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('../data/ImageNet/val',
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                 ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        print('No valid dataset is specified')
    return input_dim, input_ch, num_classes, train_loader, test_loader, dataset_train, kwargs
