from .LLBConv import LocalLossBlockConv
from .LLBLiner import LocalLossBlockLinear
from .MLP import Net
from .Resnet import ResNet
from .WResnet import Wide_ResNet
from .VGG import VGGn


def get_model(args, input_dim, input_ch, num_classes):
    if args.model == 'mlp':
        model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes, args)
    elif args.model.startswith('vgg'):
        model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult, args)
    elif args.model == 'resnet18':
        model = ResNet('basic', [2, 2, 2, 2], num_classes, input_ch, args.feat_mult, input_dim, args)
    elif args.model == 'resnet34':
        model = ResNet('basic', [3, 4, 6, 3], num_classes, input_ch, args.feat_mult, input_dim, args)
    elif args.model == 'resnet50':
        model = ResNet('bottleneck', [3, 4, 6, 3], num_classes, input_ch, args.feat_mult, input_dim, args)
    elif args.model == 'resnet101':
        model = ResNet('bottleneck', [3, 4, 23, 3], num_classes, input_ch, args.feat_mult, input_dim, args)
    elif args.model == 'resnet152':
        model = ResNet('bottleneck', [3, 8, 36, 3], num_classes, input_ch, args.feat_mult, input_dim, args)
    elif args.model == 'wresnet10-8':
        model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim, args)
    elif args.model == 'wresnet10-8a':
        model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim, True, args)
    elif args.model == 'wresnet16-4':
        model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim, args)
    elif args.model == 'wresnet16-4a':
        model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim, True, args)
    elif args.model == 'wresnet16-8':
        model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim, args)
    elif args.model == 'wresnet16-8a':
        model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim, True, args)
    elif args.model == 'wresnet28-10':
        model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim, args)
    elif args.model == 'wresnet28-10a':
        model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim, True, args)
    elif args.model == 'wresnet40-10':
        model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim)
    elif args.model == 'wresnet40-10a':
        model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim, True, args)
    else:
        print('No valid model defined')
    return model
