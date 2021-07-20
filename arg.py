import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch local error training')
    parser.add_argument('-a', '--abcde', nargs='+', default=[0.0, 0.0, 0.01, 0.99],
                        help='coefficients for losses [recon, unsup_sim, sup_sim, pred, mahi]')
    parser.add_argument('--backprop', action='store_true', default=False,
                        help='disable local loss training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--beta', type=float, default=0.99,
                        help='fraction of similarity matching loss in predsim loss (default: 0.99)')
    parser.add_argument('--bio', action='store_true', default=False,
                        help='use more biologically plausible versions of pred and sim loss (default: False)')
    parser.add_argument('--classes-per-batch', type=int, default=0,
                        help='aim for this number of different classes per batch during training (default: 0, random batches)')
    parser.add_argument('--classes-per-batch-until-epoch', type=int, default=0,
                        help='limit number of classes per batch until this epoch (default: 0, until end of training)')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout regularization')
    parser.add_argument('--dataset', default='CIFAR10',
                        help='dataset, MNIST, KuzushijiMNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, STL10 or ImageNet (default: CIFAR10)')
    parser.add_argument('--dim-in-decoder', type=int, default=4096,
                        help='input dimension of decoder_y used in pred and predsim loss (default: 4096)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout after each nonlinearity (default: 0.0)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--feat-mult', type=float, default=1,
                        help='multiply number of CNN features with this number (default: 1)')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the cutout holes in pixels')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate (default: 5e-4)')
    parser.add_argument('--loss-sup', default='predsim',
                        help='supervised local loss, sim or pred (default: predsim)')
    parser.add_argument('--loss-unsup', default='none',
                        help='unsupervised local loss, none, sim or recon (default: none)')
    parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[200, 300, 350, 375],
                        help='decay learning rate at these milestone epochs (default: [200,300,350,375])')
    parser.add_argument('--lr-decay-fact', type=float, default=0.25,
                        help='learning rate decay factor to use at milestone epochs (default: 0.25)')
    parser.add_argument('--model', default='mlp',
                        help='model, mlp, vgg13, vgg16, vgg19, vgg8b, vgg11b, resnet18, resnet34, wresnet28-10 and more (default: vgg8b)')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--no-batch-norm', action='store_true', default=False,
                        help='disable batch norm before non-linearities')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--no-detach', action='store_true', default=False,
                        help='do not detach computational graph')
    parser.add_argument('--no-print-stats', action='store_true', default=False,
                        help='do not print layerwise statistics during training with local loss')
    parser.add_argument('--no-similarity-std', action='store_true', default=False,
                        help='disable use of standard deviation in similarity matrix for feature maps')
    parser.add_argument('--nonlin', default='relu',
                        help='nonlinearity, relu or leakyrelu (default: relu)')
    parser.add_argument('--num-hidden', type=int, default=1024,
                        help='number of hidden units for mpl model (default: 1024)')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='number of hidden fully-connected layers for mlp and vgg models (default: 1')
    parser.add_argument('--optim', default='adam',
                        help='optimizer, adam, amsgrad or sgd (default: adam)')
    parser.add_argument('--pre-act', action='store_true', default=False,
                        help='use pre-activation in ResNet')
    parser.add_argument('--progress-bar', action='store_true', default=False,
                        help='show progress bar during training')
    parser.add_argument('--resume', default='', type=str,
                        help='checkpoint to resume training from')
    parser.add_argument('--save-dir', default='./results/local-error', type=str,
                        help='the directory used to save the trained models')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--target-proj-size', type=int, default=128,
                        help='size of target projection back to hidden layers for biologically plausible loss (default: 128')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    return parser.parse_args()
