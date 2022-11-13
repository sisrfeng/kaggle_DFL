if 'import':
    import os
    from datetime import datetime
    import time
    import numpy as np
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import torch


if 'local import':
    from DFL   import  Clip_of_gameS
    from model import Model
    from test import  test, testSpotting

opJ = os.path.join


if 'parse':
    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--SPLIT',       default='Test'   )
    parser.add_argument('--P_input_feat')

    parser.add_argument('--max_epochs',    type=int,   default=1000,     help='Maximum number of epochs' )
    parser.add_argument('--load_weights',       default=None,     help='weights to load' )

    parser.add_argument('--version',  type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--feature_dim',  type=int,   default=None,     help='Number of input features' )
    parser.add_argument('--evaluation_frequency',  type=int,   default=10,     help='Number of chunks per epoch' )
    parser.add_argument('--input_FpS',  type=int,   default=25 )
    parser.add_argument('--vocab_size',        type=int,   default=64, help='Size of the vocabulary for NetVLAD' )
    parser.add_argument('--NMS_threshold',        type=float,   default=0.0, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size',  type=int,   default=256,     help='Batch size' )
    parser.add_argument('--LR',        type=float,   default=1e-03, help='Learning Rate' )
    parser.add_argument('--LRe',        type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience',  type=int,   default=10,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',         type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',    type=int,   default=4, help='number of worker to load data')
    parser.add_argument('--seed',    type=int,   default=0, help='seed for reproducibility')


    parser.add_argument('--pool_win_frms_radius',  type=int )
    parser.add_argument('--pool_type',  default="NetVLAD++")

    args = parser.parse_args()

    args.model_name = args.pool_type

    if 'todo: 调参':
        args.NMS_secs3event = [ 2, 20, 6 ]
             #三类的时间窗口，play，throwin, challenge

if __name__ == '__main__':

    # for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    os.makedirs( opJ("out", args.model_name), exist_ok=True)


    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)



    start=time.time()


    # ------------------------------------main

    dataset_gameS  = Clip_of_gameS( P_feat               = args.P_input_feat         ,
                                    SPLIT                = args.SPLIT                ,
                                    input_FpS            = args.input_FpS            ,
                                    pool_win_frms_radius = args.pool_win_frms_radius ,
                                   )

    loader = torch.utils.data.DataLoader(dataset_gameS,
                                         batch_size  = 1     ,
                                         shuffle     = False ,
                                         num_workers = 1     ,
                                         pin_memory  = True  ,
                                        )

    if args.feature_dim is None:
        args.feature_dim = dataset_gameS[0][1].shape[-1]

    # create model
    model = Model(weights       = args.load_weights                 ,
                  input_size    = args.feature_dim                  ,
                  num_classes   = dataset_gameS.num_classes         ,
                  pool_win_frms = args.pool_win_frms_radius * 2 + 1 ,
                  vocab_size    = args.vocab_size                   ,
                  input_FpS     = args.input_FpS                    ,
                  pool          = args.pool_type                    ,
                 ).cuda()

    ckpt = torch.load( os.path.join("weightS", args.model_name, "best.pth")  )
    model.load_state_dict(ckpt['state_dict'])


    probS__all_vdo = testSpotting(loader,
                                  model          = model               ,
                                  model_name     = args.model_name     ,
                                  NMS_secs3event = args.NMS_secs3event ,
                                  NMS_threshold  = args.NMS_threshold  ,
                                  SPLIT          = args.SPLIT          ,
                                 )
    # ------------------------------------main

    print(f'池化头部:  {time.time() - start} 秒')
