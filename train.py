import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.backends import cudnn
from bisect import bisect_right
import math
import os
from utils import Cutout, count_parameters, to_one_hot, similarity_matrix
from models import *
from settings import parse_args
import wandb


args = parse_args()
wandb.init(config=args, project="local-loss")

if args.cuda:
    cudnn.enabled = True
    cudnn.benchmark = True
        
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

   
class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]
    
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
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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
    dataset_train = datasets.FashionMNIST('../data/FashionMNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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
    train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    dataset_train = datasets.STL10('../data/STL10', split='train', download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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
        sampler = None,
        batch_size=args.batch_size, shuffle=True, **kwargs)
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


       
if args.model == 'mlp':
    model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes)
elif args.model.startswith('vgg'):
    model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult)
elif args.model == 'resnet18':
    model = ResNet(BasicBlock, [2,2,2,2], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet34':
    model = ResNet(BasicBlock, [3,4,6,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet50':
    model = ResNet(Bottleneck, [3,4,6,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet101':
    model = ResNet(Bottleneck, [3,4,23,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet152':
    model = ResNet(Bottleneck, [3,8,36,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'wresnet10-8':
    model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet10-8a':
    model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet16-4':
    model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet16-4a':
    model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet16-8':
    model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet16-8a':
    model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet28-10':
    model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet28-10a':
    model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet40-10':
    model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet40-10a':
    model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim, True)
else:
    print('No valid model defined')
 
if args.cuda:
    model.cuda()

wandb.watch(model)

if args.progress_bar:
    from tqdm import tqdm
    
if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
elif args.optim == 'adam' or args.optim == 'amsgrad':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
else:
    print('Unknown optimizer')

model.set_learning_rate(args.lr)
print(model)
print('Model {} has {} parameters influenced by global loss'.format(args.model, count_parameters(model)))

def train(epoch, lr):
    ''' Train model on train set'''
    model.train()
    correct = 0
    loss_total_local = 0
    loss_total_global = 0
    
    # Add progress bar
    if args.progress_bar:
        pbar = tqdm(total=len(train_loader))
        
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()
                
    # Loop train set
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
  
        # Clear accumulated gradient
        optimizer.zero_grad()
        model.optim_zero_grad()
                    
        output, loss = model(data, target, target_onehot)
        loss_total_local += loss * data.size(0)
        loss = F.cross_entropy(output, target)
        if args.loss_sup == 'predsim' and not args.backprop:
            loss *= (1 - args.beta) 
        loss_total_global += loss.item() * data.size(0)
             
        # Backward pass and optimizer step
        # For local loss functions, this will only affect output layer
        loss.backward()
        optimizer.step()
        
        # If special option for no detaching is set, update weights also in hidden layers
        if args.no_detach:
            model.optim_step()
        
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()
        
        # Update progress bar
        if args.progress_bar:
            pbar.set_postfix(loss=loss.item(), refresh=False)
            pbar.update()
            
    if args.progress_bar:
        pbar.close()
        
    # Format and print debug string
    loss_average_local = loss_total_local / len(train_loader.dataset)
    loss_average_global = loss_total_global / len(train_loader.dataset)
    error_percent = 100 - 100.0 * float(correct) / len(train_loader.dataset)
    string_print = 'Train epoch={}, lr={:.2e}, loss_local={:.4f}, loss_global={:.4f}, error={:.3f}%, mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(
        epoch,
        lr, 
        loss_average_local,
        loss_average_global,
        error_percent,
        torch.cuda.memory_allocated()/1e6,
        torch.cuda.max_memory_allocated()/1e6)
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                string_print += m.print_stats() 
    print(string_print)
    
    return loss_average_local+loss_average_global, error_percent, string_print
                   
def test(epoch):
    ''' Run model on test set '''
    model.eval()
    test_loss = 0
    correct = 0
    
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()
    
    # Loop test set
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
        
        with torch.no_grad():
            output, _ = model(data, target, target_onehot)
            test_loss += F.cross_entropy(output, target).item() * data.size(0)
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()

    # Format and print debug string
    loss_average = test_loss / len(test_loader.dataset)
    if args.loss_sup == 'predsim' and not args.backprop:
        loss_average *= (1 - args.beta)
    error_percent = 100 - 100.0 * float(correct) / len(test_loader.dataset)
    string_print = 'Test loss_global={:.4f}, error={:.3f}%\n'.format(loss_average, error_percent)
    wandb.log({"Test Loss Global": loss_average, "Error": error_percent})
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                string_print += m.print_stats()                
    print(string_print)
    
    return loss_average, error_percent, string_print

''' The main training and testing loop '''

start_epoch = 1 
for epoch in range(start_epoch, args.epochs + 1):
    # Decide learning rate
    lr = args.lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch-1))
    save_state_dict = False
    for ms in args.lr_decay_milestones:
        if (epoch-1) == ms:
            print('Decaying learning rate to {}'.format(lr))
            decay = True
        elif epoch == ms:
            save_state_dict = True

    # Set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    model.set_learning_rate(lr)
    
    # Train and test    
    train_loss,train_error,train_print = train(epoch, lr)
    test_loss,test_error,test_print = test(epoch)

