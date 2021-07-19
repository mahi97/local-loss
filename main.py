import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from bisect import bisect_right
import os

from arg import get_args
from datasets import get_dataset, NClassRandomSampler

from network import *
from learner.solo_learner import SoloLearner
from learner.mix_learner import MixLearner

from tqdm import tqdm
import wandb


def count_parameters(model):
    """ Count number of parameters in model influenced by global loss. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    args = get_args()
    wandb.init(config=args, project="local-loss")
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        cudnn.enabled = True
        cudnn.benchmark = True

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    checkpoint = None
    args_backup = None
    if not args.resume == '':
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.model = checkpoint['args'].model
            args_backup = args
            args = checkpoint['args']
            args.optim = args_backup.optim
            args.momentum = args_backup.momentum
            args.weight_decay = args_backup.weight_decay
            args.dropout = args_backup.dropout
            args.no_batch_norm = args_backup.no_batch_norm
            args.cutout = args_backup.cutout
            args.length = args_backup.length
            print('=> loaded checkpoint "{}" (epoch {})'.format(args.resume, checkpoint['epoch']))
        else:
            print('Checkpoint not found: {}'.format(args.resume))

    input_dim, input_ch, num_classes, train_loader, test_loader, dataset_train, kwargs = get_dataset(args)

    model = get_model(args, input_dim, input_ch, num_classes)
    # Check if to load model
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        args = args_backup

    if args.cuda:
        model.cuda()
    wandb.watch(model)

    optimizer = None
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim == 'adam' or args.optim == 'amsgrad':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                               amsgrad=args.optim == 'amsgrad')
    else:
        print('Unknown optimizer')

    model.set_learning_rate(args.lr)
    print(model)
    print('Model {} has {} parameters influenced by global loss'.format(args.model, count_parameters(model)))

    ''' The main training and testing loop '''
    start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']
    # Train and test
    learner = SoloLearner(optimizer, args, num_classes, train_loader, test_loader)
    # learner = MixLearner(args, num_classes, train_loader, test_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        # Decide learning rate
        lr = args.lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch - 1))
        save_state_dict = False
        for ms in args.lr_decay_milestones:
            if (epoch - 1) == ms:
                print('Decaying learning rate to {}'.format(lr))
                decay = True
            elif epoch == ms:
                save_state_dict = True

        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.set_learning_rate(lr)

        # Check if to remove NClassRandomSampler from train_loader
        if 0 < args.classes_per_batch_until_epoch < epoch and isinstance(
                train_loader.sampler, NClassRandomSampler):
            print('Remove NClassRandomSampler from train_loader')
            train_loader = torch.utils.data.DataLoader(dataset_train, sampler=None, batch_size=args.batch_size,
                                                       shuffle=True, **kwargs)

        train_loss, train_error, train_print = learner.train(epoch, lr, model)
        test_loss, test_error, test_print = learner.test(epoch, model)

        # Check if to save checkpoint
        if args.save_dir is not '':
            # Resolve log folder and checkpoint file name
            filename = 'chkp_ep{}_lr{:.2e}_trainloss{:.2f}_testloss{:.2f}_trainerr{:.2f}_testerr{:.2f}.tar'.format(
                epoch, lr, train_loss, test_loss, train_error, test_error)
            dirname = os.path.join(args.save_dir, args.dataset)
            dirname = os.path.join(dirname, '{}_mult{:.1f}'.format(args.model, args.feat_mult))
            dirname = os.path.join(dirname,
                                   '{}_{}x{}_{}_{}_dimdec{}_alpha{}_beta{}_bs{}_cpb{}_drop{}{}_bn{}_{}_wd{}_bp{}_detach{}_lr{:.2e}'.format(
                                       args.nonlin, args.num_layers, args.num_hidden,
                                       args.loss_sup + '-bio' if args.bio else args.loss_sup, args.loss_unsup,
                                       args.dim_in_decoder, args.alpha,
                                       args.beta, args.batch_size, args.classes_per_batch, args.dropout,
                                       '_cutout{}x{}'.format(args.n_holes, args.length) if args.cutout else '',
                                       int(not args.no_batch_norm), args.optim, args.weight_decay, int(args.backprop),
                                       int(not args.no_detach), args.lr))

            # Create log directory
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            elif epoch == 1 and os.path.exists(dirname):
                # Delete old files
                for f in os.listdir(dirname):
                    os.remove(os.path.join(dirname, f))

            # Add log entry to log file
            with open(os.path.join(dirname, 'log.txt'), 'a') as f:
                if epoch == 1:
                    f.write('{}\n\n'.format(args))
                    f.write('{}\n\n'.format(model))
                    f.write('{}\n\n'.format(optimizer))
                    f.write('Model {} has {} parameters influenced by global loss\n\n'.format(args.model,
                                                                                              count_parameters(model)))
                f.write(train_print)
                f.write(test_print)
                f.write('\n')
                f.close()

            # Save checkpoint for every epoch
            torch.save({
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict() if (save_state_dict or epoch == args.epochs) else None,
                'train_loss': train_error,
                'train_error': train_error,
                'test_loss': test_loss,
                'test_error': test_error,
            }, os.path.join(dirname, filename))

            # Save checkpoint for last epoch with state_dict (for resuming)
            torch.save({
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'train_loss': train_error,
                'train_error': train_error,
                'test_loss': test_loss,
                'test_error': test_error,
            }, os.path.join(dirname, 'chkp_last_epoch.tar'))
