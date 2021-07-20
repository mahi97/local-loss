from .LLBConv import LocalLossBlockConv
from .LLBLiner import LocalLossBlockLinear
from .MLP import Net
from .Resnet import ResNet
from .WResnet import WideResNet
from .VGG import VGGn
from .Mahi import MAHI


def get_model(args, input_dim, input_ch, num_classes, dim_in_decoder):
    if args.model == 'mlp':
        model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes, args)
    elif args.model == 'mahi':
        model = MAHI(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes, args)
    elif args.model.startswith('vgg'):
        model = VGGn(args.model, input_dim, input_ch, num_classes, dim_in_decoder, args.feat_mult, args)
    elif args.model == 'resnet18':
        model = ResNet('basic', [2, 2, 2, 2], num_classes, input_ch, args.feat_mult, input_dim, dim_in_decoder, args)
    elif args.model == 'resnet34':
        model = ResNet('basic', [3, 4, 6, 3], num_classes, input_ch, args.feat_mult, input_dim, dim_in_decoder, args)
    elif args.model == 'resnet50':
        model = ResNet('bottleneck', [3, 4, 6, 3], num_classes, input_ch, args.feat_mult, input_dim, dim_in_decoder, args)
    elif args.model == 'resnet101':
        model = ResNet('bottleneck', [3, 4, 23, 3], num_classes, input_ch, args.feat_mult, input_dim, dim_in_decoder, args)
    elif args.model == 'resnet152':
        model = ResNet('bottleneck', [3, 8, 36, 3], num_classes, input_ch, args.feat_mult, input_dim, dim_in_decoder, args)
    elif args.model == 'wresnet10-8':
        model = WideResNet(10, 8, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args)
    elif args.model == 'wresnet10-8a':
        model = WideResNet(10, 8, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args, True)
    elif args.model == 'wresnet16-4':
        model = WideResNet(16, 4, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args)
    elif args.model == 'wresnet16-4a':
        model = WideResNet(16, 4, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args, True)
    elif args.model == 'wresnet16-8':
        model = WideResNet(16, 8, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args)
    elif args.model == 'wresnet16-8a':
        model = WideResNet(16, 8, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args, True)
    elif args.model == 'wresnet28-10':
        model = WideResNet(28, 10, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args)
    elif args.model == 'wresnet28-10a':
        model = WideResNet(28, 10, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, True, args)
    elif args.model == 'wresnet40-10':
        model = WideResNet(40, 10, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args)
    elif args.model == 'wresnet40-10a':
        model = WideResNet(40, 10, args.dropout, num_classes, input_ch, input_dim, dim_in_decoder, args, True)
    else:
        print('No valid model defined')
    return model
