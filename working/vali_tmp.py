#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and
easily modifiable ImageNet validation script for evaluating pretrained models or
training checkpoints against ImageNet or
similarly organized image datasets.

It prioritizes canonical PyTorch,  standard Python style,
and good performance.

Repurpose as you see fit.
"""

if 'import':
    import argparse
    import os
    import csv
    import glob
    import json
    import time
    import logging
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    from collections import OrderedDict
    from contextlib import suppress
    from functools import partial

    from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models, set_fast_norm
    from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
    from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser,\
        decay_batch_step, check_batch_size_retry

    has_apex = False
    try:
        from apex import amp
        has_apex = True
    except ImportError:
        pass

    has_native_amp = False
    try:
        if getattr(torch.cuda.amp, 'autocast') is not None:
            has_native_amp = True
    except AttributeError:
        pass

    try:
        from functorch.compile import memory_efficient_fusion
        has_functorch = True
    except ImportError as e:
        has_functorch = False

_logger = logging.getLogger('validate')

if 'args':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--split', metavar='NAME', default='validation',
                        help='dataset split (default: validation)')
    parser.add_argument('--dataset_download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                        help='model architecture (default: dpn92)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--img_size', default=None, type=int,
                        metavar='N', help='Input image dimension, uses model default if empty')
    parser.add_argument('--input_size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input_size 3 224 224), uses model default if empty')
    parser.add_argument('--use_train_size', action='store_true', default=False,
                        help='force use of train input size, even when test size is specified in pretrained cfg')
    parser.add_argument('--crop_pct', default=None, type=float,
                        metavar='N', help='Input image center crop pct')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number classes in dataset')
    parser.add_argument('--class_map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--log_freq', default=10, type=int,
                        metavar='N', help='batch logging frequency (default: 10)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='Number of GPUS to use')
    parser.add_argument('--test_pool', dest='test_pool', action='store_true',
                        help='enable test time pool')
    parser.add_argument('--no_prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--pin_mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--channels_last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device (accelerator) to use.")
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--amp_dtype', default='float16', type=str,
                        help='lower precision AMP dtype (default: float16)')
    parser.add_argument('--amp_impl', default='native', type=str,
                        help='AMP impl to use, "native" or "apex" (default: native)')
    parser.add_argument('--tf_preprocessing', action='store_true', default=False,
                        help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
    parser.add_argument('--use_ema', dest='use_ema', action='store_true',
                        help='use ema version of weights if present')
    scripting_group = parser.add_mutually_exclusive_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='torch.jit.script the full model')
    scripting_group.add_argument('--aot_autograd', default=False, action='store_true',
                        help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
    parser.add_argument('--fuser', default='', type=str,
                        help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    parser.add_argument('--fast_norm', default=False, action='store_true',
                        help='enable experimental fast-norm')
    parser.add_argument('--results_file', default='', type=str, metavar='FILENAME',
                        help='Output csv file for validation results (summary)')
    parser.add_argument('--real_labels', default='', type=str, metavar='FILENAME',
                        help='Real labels JSON file for imagenet evaluation')
    parser.add_argument('--valid_labels', default='', type=str, metavar='FILENAME',
                        help='Valid label indices txt file for validation of partial label space')
    parser.add_argument('--retry', default=False, action='store_true',
                        help='Enable batch size decay & retry for single model validation')


def validate(args):
    # might as well try to validate something
    if 'handle args':
        args.pretrained = args.pretrained or  not args.checkpoint
        args.prefetcher = not args.no_prefetcher

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        device = torch.device(args.device)

        # resolve AMP arguments based on PyTorch / Apex availability
        use_amp = None
        amp_autocast = suppress
        if args.amp:
            if args.amp_impl == 'apex':
                assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
                assert args.amp_dtype == 'float16'
                use_amp = 'apex'
                _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
            else:
                assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
                assert args.amp_dtype in ('float16', 'bfloat16')
                use_amp = 'native'
                amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
                amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
                _logger.info('Validating in mixed precision with native PyTorch AMP.')
        else:
            _logger.info('Validating in float32. AMP not enabled.')

        if args.fuser:
            set_jit_fuser(args.fuser)

        if args.fast_norm:
            set_fast_norm()

    # create model
    model = create_model(
        args.model                     ,
        pretrained  = args.pretrained  ,
        num_classes = args.num_classes ,
        in_chans    = 3                ,
        global_pool = args.gp          ,
        scriptable  = args.torchscript ,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    # param_count = sum([m.numel() for m in model.parameters()])
    # _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args)                              ,
        model         = model                   ,
        use_test_size = not args.use_train_size ,
        verbose       = True                    ,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot_autograd"
        model = memory_efficient_fusion(model)

    model = model.to(device)
    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().to(device)

    dataset = create_dataset(
        root       = args.data             ,
        name       = args.dataset          ,
        split      = args.split            ,
        download   = args.dataset_download ,
        load_bytes = args.tf_preprocessing ,
        class_map  = args.class_map        ,
    )

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset                                         ,
        input_size       = data_config['input_size']    ,
        batch_size       = args.batch_size              ,
        use_prefetcher   = args.prefetcher              ,
        interpolation    = data_config['interpolation'] ,
        mean             = data_config['mean']          ,
        std              = data_config['std']           ,
        num_workers      = args.workers                 ,
        crop_pct         = crop_pct                     ,
        pin_memory       = args.pin_mem                 ,
        device           = device                       ,
        tf_preprocessing = args.tf_preprocessing        ,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model         = args.model                    ,
        top1          = round(top1a, 4)               ,
        top5          = round(top5a, 4)               ,
        top1_err      = round(100 - top1a, 4)         ,
        top5_err      = round(100 - top5a, 4)         ,
        param_count   = round(param_count / 1e6, 2)   ,
        img_size      = data_config['input_size'][-1] ,
        crop_pct      = crop_pct                      ,
        interpolation = data_config['interpolation']  ,
    )

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(results['top1'],
                                                                         results['top1_err'],
                                                                         results['top5'],
                                                                         results['top5_err'],
                                                                        ))

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results    = OrderedDict()
    error_str  = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if torch.cuda.is_available() and  \
              'cuda' in args.device:
                torch.cuda.empty_cache()
            # 调用validate()
            return  validate(args)
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')

    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


def main():
    setup_default_logging()
    args        = parser.parse_args()
    model_cfgs  = []
    model_names = []

    if os.path.isdir(args.checkpoint):
        # validate ¿all checkpoints¿ in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names
            # with pretrained checkpoints
            args.pretrained = True
            model_names     = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k', '*_dino'])
            model_cfgs      = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and  \
          os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]


    if len(model_cfgs):
        results_file = args.results_file or  \
                      './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results,
                         key     = lambda x: x['top1'],
                         reverse = True,
                        )
        if len(results):
            write_results(results_file, results)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)
    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf,
                            fieldnames=results[0].keys(),
                           )
        dw.writeheader()
        for r in results:
            dw.writerow(r)

        cf.flush()


if __name__ == '__main__':
    main()
