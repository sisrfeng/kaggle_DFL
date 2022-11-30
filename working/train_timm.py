# 此文件 本来在timm的github repo, 不在pip的安装目录
# 这导致pudb调用/usr下的python:  ¿#!/usr/bin/env python3¿
"""
ImageNet Training Script

    This is intended to be a lean and
    easily modifiable ImageNet training script that reproduces ImageNet training results with some of the latest networks and
    training techniques.
    It favours canonical PyTorch and
    standard Python style over
        trying to be able to 'do it all.' That said,
    it offers quite a few speed and
                            training result
                        improvements over the usual PyTorch example  scripts.
    Repurpose as you see fit.

    This script was started from an early version of
        the PyTorch ImageNet example  (https://github.com/pytorch/examples/tree/master/imagenet)

    NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
    (https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

    Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
if 1:
    from ap_score import export_ap_score
    import numpy as np
    import argparse
    import logging
    import os
    import time
    from collections import OrderedDict
    from contextlib import suppress
    from datetime import datetime

    import torch
    import torch.nn as nn
    import torchvision.utils
    import yaml
    from torch.nn.parallel import DistributedDataParallel as NativeDDP

    import sys
    sys.path.insert(1, '../input/upload-wf')

    from timm import utils
    from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
    from timm.loss import JsdCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
    from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
                            convert_splitbn_model, convert_sync_batchnorm, model_parameters, set_fast_norm
    from timm.optim import create_optimizer_v2, optimizer_kwargs
    from timm.scheduler import create_scheduler
    from timm.utils import ApexScaler, NativeScaler

    try:
        from apex import amp
        from apex.parallel import DistributedDataParallel as ApexDDP
        from apex.parallel import convert_syncbn_model
        has_apex = True
    except ImportError:
        has_apex = False

    has_native_amp = False
    try:
        if getattr(torch.cuda.amp, 'autocast') is not None:
            has_native_amp = True
    except AttributeError:
        pass

    try:
        import wandb
        has_wandb = True
    except ImportError:
        has_wandb = False

    try:
        from functorch.compile import memory_efficient_fusion
        has_functorch = True
    except ImportError as e:
        has_functorch = False


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')  # 现在这_logger 只print, 不存到文件  要指定filter? stream?

# The first arg parser parses out
# only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser()
if 'arg':
    # Keep this argument outside of the dataset group
    # because it is positional.
    parser.add_argument('data_dir', metavar='DIR', default='../work/imgs4train')
        # imgstrain/val, 不包含整段验证视频
        # 在label中, 上一个事件的end和下一个事件的start 之间的帧, 算eventAP时被忽略, 不论预测为什么都不影响得分
            # 在比赛test set上推理时, 没有label, 所以每一帧都要推理,
            # 但在训练过程中验证时, 有label, 没必要管那些不算分的帧

    # parser.add_argument('--data_dir', metavar='DIR', default='../work/imgs4train')

    if 'Dataset parameters':
        group = parser.add_argument_group('Dataset parameters')
        group.add_argument('--dataset', '-d', default='',
                            help='dataset type (default: ImageFolder/ImageTar if empty)')
        group.add_argument('--train_split', default='train' )
        group.add_argument('--val_split', default='validation',
                            help='dataset validation split (default: validation)')
        group.add_argument('--dataset_download', action='store_true', default=False,
                            help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
        group.add_argument('--class_map', default='', metavar='FILENAME',
                            help='path to class to idx mapping file (default: "")')

    if 'Model parameters':
        group = parser.add_argument_group('Model parameters')
        group.add_argument('--model', default='tf_efficientnet_b5_ap', metavar='MODEL',  help='Name of model to train (default: "resnet50"')

        group.add_argument('--pretrained', action='store_true', default=False,  help='Start with pretrained version of specified network (if avail)')
        # 放着, 好搜:--checkpoint --load_checkpoint
        group.add_argument('--initial_checkpoint', default='', metavar='PATH',
                           help='Initialize model from this checkpoint (default: none), 会覆盖上面的--pretrained?')
        group.add_argument('--resume', default="", metavar='PATH',  help='Resume full model and optimizer state from checkpoint (default: none)')

        group.add_argument('--no_resume_opt', action='store_true', default=False,
                            help='prevent resume of optimizer state when resuming model')
        group.add_argument('--num_classes', type=int, default=4, metavar='N',
                            help='number of label classes (Model default if None)')
        group.add_argument('--gp', default=None, metavar='POOL',
                            help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
        group.add_argument('--img_size', type=int, default=None, metavar='N',
                            help='Image patch size (default: None => model default)')
        group.add_argument('--input_size', default=None, nargs=3, type=int,
                            metavar='N N N', help='Input all image dimensions (d h w, e.g. --input_size 3 224 224), uses model default if empty')
        group.add_argument('--crop_pct', default=None, type=float,
                            metavar='N', help='Input image center crop percent (for validation only)')
        group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                            help='Override mean pixel value of dataset')
        group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                            help='Override std deviation of dataset')
        group.add_argument('--interpolation', default='',
                            help='Image resize interpolation type (overrides model)')

        group.add_argument('-b', '--batch_size', type=int, default=4, metavar='N',
                            help='Input batch size for training (default: 128), 我改成了4')
        group.add_argument('-vb', '--validation_batch_size', type=int, default=None, metavar='N',
                            help='Validation batch size override (default: None)')
        group.add_argument('--channels_last', action='store_true', default=False,
                            help='Use channels_last memory layout')
        scripting_group = group.add_mutually_exclusive_group()
        scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                            help='torch.jit.script the full model')
        scripting_group.add_argument('--aot_autograd', default=False, action='store_true',
                                help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
        group.add_argument('--fuser', default='',
                            help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
        group.add_argument('--fast_norm', default=False, action='store_true',
                            help='enable experimental fast-norm')
        group.add_argument('--grad_checkpointing', action='store_true', default=False,
                            help='Enable gradient checkpointing through model blocks/stages')

    group = parser.add_argument_group('Optimizer parameters')
    if 1:
        group.add_argument('--opt', default='sgd', metavar='OPTIMIZER',
                            help='Optimizer (default: "sgd"')
        group.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: None, use opt default)')
        group.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: None, use opt default)')
        group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='Optimizer momentum (default: 0.9)')
        group.add_argument('--weight_decay', type=float, default=2e-5,
                            help='weight decay (default: 2e-5)')
        group.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                            help='Clip gradient norm (default: None, no clipping)')
        group.add_argument('--clip_mode', default='norm',
                            help='Gradient clipping mode. One of ("norm", "value", "agc")')
        group.add_argument('--layer_decay', type=float, default=None,
                            help='layer-wise learning rate decay (default: None)')

    if  'Learning rate schedule parameters':
        group = parser.add_argument_group('Learning rate schedule parameters')
        group.add_argument('--sched', default='cosine', metavar='SCHEDULER',
                            help='LR scheduler (default: "step"')
        group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                            help='learning rate (default: 0.05)')
        group.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                            help='learning rate noise on/off epoch percentages')
        group.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                            help='learning rate noise limit percent (default: 0.67)')
        group.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                            help='learning rate noise std-dev (default: 1.0)')
        group.add_argument('--lr_cycle_mul', type=float, default=1.0, metavar='MULT',
                            help='learning rate cycle len multiplier (default: 1.0)')
        group.add_argument('--lr_cycle_decay', type=float, default=0.5, metavar='MULT',
                            help='amount to decay each learning rate cycle (default: 0.5)')
        group.add_argument('--lr_cycle_limit', type=int, default=1, metavar='N',
                            help='learning rate cycle limit, cycles enabled if > 1')
        group.add_argument('--lr_k_decay', type=float, default=1.0,
                            help='learning rate k-decay for cosine/poly (default: 1.0)')
        group.add_argument('--warmup_lr', type=float, default=0.0001, metavar='LR',
                            help='warmup learning rate (default: 0.0001)')
        group.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
        group.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='number of epochs to train (default: 300)')
        group.add_argument('--epoch_repeats', type=float, default=0., metavar='N',
                            help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
        group.add_argument('--start_epoch', default=None, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        group.add_argument('--decay_milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                            help='list of decay epoch indices for multistep lr. must be increasing')
        group.add_argument('--decay_epochs', type=float, default=100, metavar='N',
                            help='epoch interval to decay LR')
        group.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
        group.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                            help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
        group.add_argument('--patience_epochs', type=int, default=10, metavar='N',
                            help='patience epochs for Plateau LR scheduler (default: 10')
        group.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                            help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    group = parser.add_argument_group('Augmentation and regularization parameters')
    group.add_argument('--no_aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    group.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    group.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    group.add_argument('--aa', default=None,
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--aug_repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    group.add_argument('--aug_splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')

    group.add_argument('--jsd_loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug_splits`.')

    group.add_argument('--bce_loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--bce_target_thresh', type=float, default=None,
                                help='Threshold for binarizing softened BCE targets (default: None, disabled)')

    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    group.add_argument('--remode', default='pixel',
                        help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup_mode', default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup_off_epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    group.add_argument('--train_interpolation', default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    group.add_argument('--drop_connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    group.add_argument('--drop_path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    group.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
    group.add_argument('--bn_momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    group.add_argument('--bn_eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    group.add_argument('--sync_bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    group.add_argument('--dist_bn', default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    group.add_argument('--split_bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    group = parser.add_argument_group('Model exponential moving average parameters')
    group.add_argument('--model_ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    group.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model_ema_decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    if 'Misc':
        group = parser.add_argument_group('Miscellaneous parameters')
        group.add_argument('--seed', type=int, default=42, metavar='S',
                            help='random seed (default: 42)')
        group.add_argument('--worker_seeding', default='all',
                            help='worker seed mode (default: all)')
        group.add_argument('--log_interval', type=int, default=5000, metavar='N',
                            help='how many batches to wait before logging training status')
        group.add_argument('--recovery_interval', type=int, default=0, metavar='N',
                            help='how many batches to wait before writing recovery checkpoint')
        group.add_argument('--checkpoint_hist', type=int, default=10, metavar='N',
                            help='number of checkpoints to keep (default: 10)')
        group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                            help='how many training processes to use (default: 4)')
        group.add_argument('--save_images', action='store_true', default=False,
                            help='save images of input bathes every log interval for debugging')

        group.add_argument('--amp', action='store_true', default=True,
                           help='use mixed precision training, 有Native AMP就用它, 没有就用NVIDIA Apex AMP')

        group.add_argument('--apex_amp', action='store_true', default=False,
                            help='指定用Use NVIDIA Apex AMP' )
        group.add_argument('--native_amp', action='store_true', default=False,
                            help='指定用Use Native Torch AMP' )

        group.add_argument('--no_ddp_bb', action='store_true', default=False,
                            help='Force broadcast buffers for native DDP to off.')
        group.add_argument('--pin_mem', action='store_true', default=False,
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        group.add_argument('--no_prefetcher', action='store_true', default=False,
                            help='disable fast prefetcher')
        group.add_argument('--output', default='./', metavar='PATH',
                            help='path to output folder (default: none, current dir)')
        group.add_argument('--experiment', default='use_bce_loss__val_on_eventAP',
                            help='name of train experiment, name of sub-folder for output')
        group.add_argument('--which_metric', default='loss',
                            help='Best metric (default: "top1"')
        group.add_argument('--tta', type=int, default=0, metavar='N',
                            help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
        group.add_argument("--local_rank", default=0, type=int)
        group.add_argument('--use_multi_epochs_loader', action='store_true', default=False,
                            help='use the multi-epochs-loader to save time at the beginning of every epoch')
        group.add_argument('--log_wandb', action='store_true', default=False,
                            help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args,
    # the usual  defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    # import pudb
    # pu.db
    utils.setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device     = 'cuda:0'
    args.world_size = 1
    args.rank       = 0  # global rank
    if args.distributed:
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.getenv('LOCAL_RANK'))
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('单GPU, 且单process')
    assert args.rank >= 0

    if args.rank == 0 and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    if 'resolve AMP arguments based on PyTorch/Apex availability':
        use_amp = None
        if args.amp:
            # `--amp` chooses native amp before apex (APEX ver: not actively maintained)
            if has_native_amp:
                args.native_amp = True
            elif has_apex:
                args.apex_amp = True

        if args.apex_amp and has_apex:
            use_amp = 'apex'
        elif args.native_amp and has_native_amp:
            use_amp = 'native'
        elif args.apex_amp or args.native_amp:
            _logger.warning("APEX 或 native Torch AMP 都没有, using float32. "
                            "Install NVIDA apex or upgrade to PyTorch 1.6")

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)

    if args.fast_norm:
        set_fast_norm()



    model = create_model(args.model,
                         in_chans        = 3                ,
                         pretrained        = 1                       ,
                         # pretrained        = args.pretrained         ,  为了用DAP来debug, 只好强行改内部代码
                         num_classes       = args.num_classes        ,
                         drop_rate         = args.drop               ,
                         drop_path_rate    = args.drop_path          ,
                         drop_block_rate   = args.drop_block         ,
                         global_pool       = args.gp                 ,
                         bn_momentum       = args.bn_momentum        ,
                         bn_eps            = args.bn_eps             ,
                         scriptable        = args.torchscript        ,
                         checkpoint_path   = args.initial_checkpoint ,
                         drop_connect_rate = args.drop_connect       , # DEPRECATED, use drop_path
                        )

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if args.local_rank == 0:  # 没用分布式训练?
        _logger.info(  f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args)                     ,
                                      model   = model                ,
                                      verbose = args.local_rank == 0 ,
                                     )

    # setup augmentation batch splits for
    # contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1, makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot_autograd"
        model = memory_efficient_fusion(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP)
    #    loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')


    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(model,
                                         args.resume,
                                         optimizer   = None if args.no_resume_opt else optimizer,
                                         loss_scaler = None if args.no_resume_opt else loss_scaler,
                                         log_info    = args.local_rank == 0,
                                        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch

    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create_dataset默认处理的是: timm folder (or tar) based ImageDataset
    # 会调用ImageDataset(1个class)
    dataset_train = create_dataset(args.dataset,
                                   root        = args.data_dir         ,
                                   split       = args.train_split      ,
                                   is_training = True                  ,
                                   class_map   = args.class_map        ,
                                   download    = args.dataset_download ,
                                   batch_size  = args.batch_size       ,
                                   repeats     = args.epoch_repeats    ,
                                  )

    dataset_eval = create_dataset(args.dataset,
                                  root        = args.data_dir         ,
                                  split       = args.val_split        ,
                                  is_training = False                 ,
                                  class_map   = args.class_map        ,
                                  download    = args.dataset_download ,
                                  batch_size  = args.batch_size       ,
                                 )

    # setup mixup or cutmix
    collate_fn = None
    mixup_fn   = None
    mixup_active = args.mixup > 0 or  \
                   args.cutmix > 0. or  \
                   args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(mixup_alpha     = args.mixup             ,
                          cutmix_alpha    = args.cutmix            ,
                          cutmix_minmax   = args.cutmix_minmax     ,
                          prob            = args.mixup_prob        ,
                          switch_prob     = args.mixup_switch_prob ,
                          mode            = args.mixup_mode        ,
                          label_smoothing = args.smoothing         ,
                          num_classes     = args.num_classes       ,
                         )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    loader_train = create_loader(
        dataset_train                                          ,
        input_size              = data_config['input_size']    ,
        batch_size              = args.batch_size              ,
        is_training             = True                         ,
        use_prefetcher          = args.prefetcher              ,
        no_aug                  = args.no_aug                  ,
        re_prob                 = args.reprob                  ,
        re_mode                 = args.remode                  ,
        re_count                = args.recount                 ,
        re_split                = args.resplit                 ,
        scale                   = args.scale                   ,
        ratio                   = args.ratio                   ,
        hflip                   = args.hflip                   ,
        vflip                   = args.vflip                   ,
        color_jitter            = args.color_jitter            ,
        auto_augment            = args.aa                      ,
        num_aug_repeats         = args.aug_repeats             ,
        num_aug_splits          = num_aug_splits               ,
        interpolation           = train_interpolation          ,
        mean                    = data_config['mean']          ,
        std                     = data_config['std']           ,
        num_workers             = args.workers                 ,
        distributed             = args.distributed             ,
        collate_fn              = collate_fn                   ,
        pin_memory              = args.pin_mem                 ,
        use_multi_epochs_loader = args.use_multi_epochs_loader ,
        worker_seeding          = args.worker_seeding          ,
    )

    loader_eval = create_loader(
        dataset_eval                                                   ,
        input_size     = data_config['input_size']                     ,
        batch_size     = args.validation_batch_size or args.batch_size ,
        is_training    = False                                         ,
        use_prefetcher = args.prefetcher                               ,
        interpolation  = data_config['interpolation']                  ,
        mean           = data_config['mean']                           ,
        std            = data_config['std']                            ,
        num_workers    = args.workers                                  ,
        distributed    = args.distributed                              ,
        crop_pct       = data_config['crop_pct']                       ,
        pin_memory     = args.pin_mem                                  ,
    )

    if 'setup loss function':
        # import pudb; pu.db
        if args.jsd_loss:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)

        elif mixup_active:
            # smoothing is handled with
            # mixup target transform which outputs sparse, soft targets
            # args.bce_loss = 1  # 我为了用DAP来debug, 强制指定
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()

        elif args.smoothing:
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh, smoothing=args.smoothing)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.cuda()
        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    which_metric = args.which_metric
    best_metric  = None
    best_epoch   = None
    saver        = None
    output_dir   = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train',
                                      exp_name
                                     )
        decreasing = True if which_metric == 'loss'  \
                          else  \
                     False
        saver = utils.CheckpointSaver(model          = model                ,
                                      optimizer      = optimizer            ,
                                      args           = args                 ,
                                      model_ema      = model_ema            ,
                                      amp_scaler     = loss_scaler          ,
                                      checkpoint_dir = output_dir           ,
                                      recovery_dir   = output_dir           ,
                                      decreasing     = decreasing           ,
                                      max_history    = args.checkpoint_hist ,
                                     )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and \
            hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(epoch,
                                            model                       ,
                                            loader_train                ,
                                            optimizer                   ,
                                            train_loss_fn               ,
                                            args                        ,
                                            lr_scheduler = lr_scheduler ,
                                            saver        = saver        ,
                                            output_dir   = output_dir   ,
                                            amp_autocast = amp_autocast ,
                                            loss_scaler  = loss_scaler  ,
                                            model_ema    = model_ema    ,
                                            mixup_fn     = mixup_fn     ,
                                           )
            # import pudb
            # pu.db

            if args.distributed and  \
              args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model,
                                    loader_eval                 ,
                                    validate_loss_fn            ,
                                    args                        ,
                                    amp_autocast = amp_autocast ,
                                   )

            if model_ema is not None and  \
            not args.model_ema_force_cpu:
                if args.distributed and \
                  args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema,
                                        args.world_size,
                                        args.dist_bn == 'reduce',
                                       )

                eval_metrics = validate(model_ema.module          ,
                                        loader_eval               ,
                                        validate_loss_fn          ,
                                        args                      ,
                                        amp_autocast=amp_autocast ,
                                        log_suffix=' (EMA)'       ,
                                       )

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch+1, eval_metrics[which_metric])

            if output_dir is not None:
                utils.update_summary(epoch                                   ,
                                     train_metrics                           ,
                                     eval_metrics                            ,
                                     os.path.join(output_dir, 'log_summary.csv') ,
                                     write_header = best_metric is None          ,
                                     log_wandb    = args.log_wandb and has_wandb ,
                                    )
                # args.log_wandb是 False
                # has_wandb是 False


            if saver is not None:
                # save proper checkpoint with eval metric
                print(f'{which_metric=}')
                best_metric, best_epoch = saver.save_checkpoint(epoch,
                                                                metric= eval_metrics[which_metric],
                                                               )

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(epoch,
                    model                   ,
                    loader                  ,
                    optimizer               ,
                    loss_fn                 ,
                    args                    ,
                    lr_scheduler = None     ,
                    saver        = None     ,
                    output_dir   = None     ,
                    amp_autocast = suppress ,
                    loss_scaler  = None     ,
                    model_ema    = None     ,
                    mixup_fn     = None     ,
                   ):

    if args.mixup_off_epoch and \
    epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and \
                   optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m  = utils.AverageMeter()
    losses_m     = utils.AverageMeter()

    model.train()

    end         = time.time()
    last_idx    = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (input, target) in enumerate(loader):
        last_batch =   batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            # 这里在train_one_epoch(), validate()里也有model(input)
            output = model(input)
            loss   = loss_fn(output ,target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss,
                        optimizer,
                        clip_grad    = args.clip_grad,
                        clip_mode    = args.clip_mode,
                        parameters   = model_parameters(model, exclude_head='agc' in args.clip_mode),
                        create_graph = second_order,
                       )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(model_parameters(model, exclude_head='agc' in args.clip_mode),
                                         value = args.clip_grad ,
                                         mode  = args.clip_mode ,
                                        )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or \
        batch_idx % args.log_interval == 0:
            LRs = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(LRs) / len(LRs)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                # _logger.info(
                    #     'epoch: {} [ batches:{} x {:>3.0f}%]  '
                    #     'Loss: {loss.val:#.3g}'
                    #     'batch_time: {batch_time.val:>4.3f}s, fps: {rate:>4.2f}  '
                    #     'LR: {lr:.1e}  '
                    #     'data_time: {data_time.val:.3f}'.format(
                    #         epoch                       ,
                    #         batch_idx                   ,
                    #         len(loader)                 ,
                    #         100. * batch_idx / last_idx ,
                    #         loss       = losses_m       ,
                    #         batch_time = batch_time_m   ,
                    #         rate       = input.size(0) * args.world_size / batch_time_m.val,
                    #         # rate_avg   = input.size(0) * args.world_size / batch_time_m.avg,
                    #         lr         = lr,
                    #         data_time  = data_time_m,
                    #     )
                    # )

                _logger.info(
                        f'{epoch=}  {len(loader)= } x {100. * batch_idx / last_idx :>3.0f}%  '
                        f'Loss: {losses_m.val:>4.2f}    '
                        f'batch_time:  {batch_time_m.val:>2.1f}秒, '
                        f'fps: {input.size(0) * args.world_size / batch_time_m.val:>4.2f}   '
                        f'LR: {lr:>3.1e}   '
                        f'data_time: {data_time_m.val*1000:>5.3f}ms'
                )


                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input                                                      ,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx) ,
                        padding   = 0                                              ,
                        normalize = True)

        if saver is not None and  \
        args.recovery_interval and \
        ( last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()


    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    losses_m     = utils.AverageMeter()
    top1_m       = utils.AverageMeter()

    model.eval()

    end      = time.time()
    last_idx = len(loader) - 1
    probS  = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            probS.append(output.cpu().numpy())

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            end = time.time()

            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: {batch_idx:>4d}/{last_idx}  '
                    f'Loss: {losses_m.val:>6.2f}   '
                    f'Acc:  {top1_m.val:>7.2f}'
                )

        #
    probS         = np.concatenate(probS, axis=0)
    filenames_val = loader.dataset.filenames(basename=True)

    event_AP =  export_ap_score(probS, filenames_val)

    metrics = OrderedDict([('loss'  ,losses_m.avg) ,
                            ('top1'     ,top1_m.avg)   ,
                            ('event_AP' ,event_AP)     ,
                            ]
                            )

    return metrics



if __name__ == '__main__':
    main()
