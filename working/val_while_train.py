# !mkdir -p /kaggle/work

SPLIT        = 'Val'
# SPLIT        = 'Test'
need_vdo2img = 0


if "import xxx":
    # !mkdir ....
    import glob
    import os
    from tqdm.auto import tqdm
    from multiprocessing import Pool, cpu_count
    import cv2
    import time
    import argparse
    import numpy as np
    import torch
    import pandas as pd

    import sys
    # ✗sys.path.append('../input/upload-wf')✗
    # ✗用pip装的✗
    sys.path.insert(1, '../input/upload-wf')

    from timm.models import create_model, apply_test_time_pool
    from timm.data   import ImageDataset, create_loader, resolve_data_config



IMG_SIZE = (456 , 456)


P_data   = '../input/dfl-bundesliga-data-shootout'

if SPLIT == 'Test':
    P_vdo = sorted(glob.glob(f'{P_data}/test/*'))
    P_img   = '../work/Test_imgs'
else:
    P_vdo = [f'{P_data}/train/3c993bd2_0.mp4',
             f'{P_data}/train/3c993bd2_1.mp4']
    P_img = '../work/Val_imgs'
    # P_img = '../work/train_imgs'

if need_vdo2img:
    def vdo2img(video_path, out_dir):
        # print(video_path)
        video_name = os.path.basename(video_path).split('.')[0]
        cap        = cv2.VideoCapture(video_path)
        frm_id = 1
        while True:
            ok, img = cap.read()
            if not ok:
                break
            outfile = f'{out_dir}/{video_name}-{frm_id:06}.jpg'
            img = cv2.resize(img, dsize=IMG_SIZE)
            cv2.imwrite(outfile, img)
            #print(outfile)
            frm_id += 1

    if not os.path.exists(P_img):
        os.system(  'mkdir -p ' + P_img )
        for video_path in P_vdo:
            vdo2img(video_path, P_img)

if 'get 2 npy, {SPLIT}_feat2048有1.5G':
    if os.path.isfile(f'./{SPLIT}_feat2048.npy') and \
      os.path.isfile(f'./{SPLIT}_nameS__vdo_img.npy'):
        print('之前保存了infer的结果')
        feat2048        = np.load(f'./{SPLIT}_feat2048.npy')
        vdo_img_nameS = np.load(f'./{SPLIT}_nameS__vdo_img.npy').tolist()
    else:
        torch.backends.cudnn.benchmark = True

        # 用一个class用来传参
        class Args:
            batch_size    = 32
            workers       = 8
            checkpoint    = '../input/weight-456/weight-456'
            data          = P_img
            img_size      = None
            input_size    = None
            interpolation = ''
            print_freq      = 1000
            mean          = None
            no_test_pool  = False
            num_classes   = 4
            num_gpu       = 1
            pretrained    = False
            std           = None
            # topk          = 5  没用的吧
            arch         = 'tf_efficientnet_b5_ap'  # ap表示AdvProp?
            # 来自/data2/wf2/s_kaggle/work/timm/pytorch-image-models/docs/models/advprop.md:
                # - Name: tf_efficientnet_b5_ap
                    #     In Collection: AdvProp
                    #     Metadata:
                    #         FLOPs      :  13176501888
                    #         Parameters :  30390000
                    #         File Size  :  122403150
                    #         Architecture:
                    #             - 1x1 Convolution
                    #             - Average Pooling
                    #             - Batch Normalization
                    #             - Convolution
                    #             - Dense Connections
                    #             - Dropout
                    #             - Inverted Residual Block
                    #             - Squeeze-and-Excitation Block
                    #             - Swish
                    #         Tasks:
                    #             - Image Classification
                    #         Training Techniques:
                    #             - AdvProp
                    #             - AutoAugment
                    #             - Label Smoothing
                    #             - RMSProp
                    #             - Stochastic Depth
                    #             - Weight Decay
                    #         Training Data:
                    #             - ImageNet
                    #         ID                 :  tf_efficientnet_b5_ap
                    #         LR                 :  0.256
                    #         Epochs             :  350
                    #         Crop Pct           :  '0.934'
                    #         Momentum           :  0.9
                    #         Batch Size         :  2048
                    #         Image Size         :  '456'
                    #         Weight Decay       :  1.0e-05
                    #         Interpolation      :  bicubic
                    #         RMSProp Decay      :  0.9
                    #         Label Smoothing    :  0.1
                    #         BatchNorm Momentum :  0.99
                    #
                    #     Code:
                    #         https://github.com/rwightman/pytorch-image-models/blob/9a25fdf3ad0414b4d66da443fe60ae0aa14edc84/timm/models/efficientnet.py#L1384
                    #     Weights:
                    #         https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth
                    #     Results:
                    #         - Task: Image Classification
                    #             Dataset: ImageNet
                    #             Metrics:
                    #                 Top 1 Accuracy: 84.25%
                    #                 Top 5 Accuracy: 96.97%


        def infer(Args):
            #Args = parser.parse_args()
            #print(Args)
            # might as well try to do something useful...
            Args.pretrained =   Args.pretrained     or  \
                                not Args.checkpoint

            # create model
            model = create_model(Args.arch,
                                num_classes     = Args.num_classes ,
                                in_chans        = 3                ,
                                pretrained      = Args.pretrained  ,
                                checkpoint_path = Args.checkpoint  ,
                                )


            loader_cfg = resolve_data_config(vars(Args), model=model)
                    # 这个函数的作用: 对于Args里未指定的, 就用timm提供的默认值

            model, test_time_pool_01 = (model, False) \
                                                if Args.no_test_pool \
                                                else  \
                                    apply_test_time_pool(model, loader_cfg)
                                    # apply_test_time_pool  返回model(原封不动), 以及False

            model.reset_classifier(0)
                  # reset_classifier(# self, num_classes, global_pool='avg')


            if Args.num_gpu > 1:
                model = torch.nn.DataParallel(model, device_ids=list(range(Args.num_gpu)) ).cuda()
            else:
                model = model.cuda()

            loader = create_loader(
                ImageDataset(Args.data),
                use_prefetcher = True                    ,
                batch_size     = Args.batch_size         ,
                num_workers    = Args.workers            ,
                # 上面几个 不需要经过resolve_data_config来wrap
                input_size     = loader_cfg['input_size']    ,
                interpolation  = loader_cfg['interpolation'] ,
                mean           = loader_cfg['mean']          ,
                std            = loader_cfg['std']           ,
                crop_pct       = loader_cfg['crop_pct'] \
                                    if not test_time_pool_01  \
                                    else \
                                 1.0  # efficientNet默认的cropt_pct是0.93左右
                )
            if loader_cfg['input_size'][-1] > 456:
                # print(f'{loader_cfg["input_size"]= }')
                pass

            model.eval()

            feat2048       = []
            with torch.no_grad():
                for batch_idx, (input, _) in enumerate(loader):
                    print(f'{batch_idx= }')
                    input  = input.cuda()
                    guess = model(input)
                    # https://www.kaggle.com/code/assign/dfl-tito-tta/notebook 的唯一改动:
                    # guess = model(input) + model(input.flip(-1))
                        # 但用了时间池化 貌似就不能直接这样?  因为后面的还有FC,
                        # train和test时不一致
                        # 在最后的sigmoid处 取flip前后平均?
                    feat2048.append( guess.cpu().numpy() )

                    if batch_idx % Args.print_freq == 0:
                        # print('Predict: [{0}/{1}] '.format(batch_idx, len(loader) ))
                        pass
            # feat2048:  a list, len为3000
            feat2048 = np.concatenate(feat2048, axis=0)
            # 对于test:  feat2048.shape: (24000, 4)   (3000 x 8, 8 是batch_size?)

            import pudb
            pu.db
            return feat2048, loader.dataset.filenames(basename=True)


        feat2048 , vdo_img_nameS = infer(Args)
                                   # Args作为一个class 不实例化就直接用, 少见

        np.save(f'./{SPLIT}_feat2048'      , feat2048)
        np.save(f'./{SPLIT}_nameS__vdo_img' , vdo_img_nameS)

if '时间池化, 每个视频保存一个npy':
    import subprocess
    sub_cmd = f'cd ../input/upload-wf/pool1D ; python use_pool.py  --SPLIT {SPLIT} --P_input_feat {SPLIT}_feat2048.npy'
    sub_cmd = sub_cmd + ' --pool_win_frms_radius 5'
    sub_fail = subprocess.Popen(sub_cmd, shell=True).wait()
    if sub_fail:
        exit()

if '汇总网络输出的多个npy':
    # probS_all_vdo =  np.load('../input/upload-wf/pool1D/probS__all_vdo.npy')

    nameS__vdo_img = []
    probS_all_vdo = []

    if SPLIT == 'Test':
        ######下面是将每个视频置信度的,npy链接在一起
        import glob
        gameS_long = sorted(glob.glob('../input/dfl-bundesliga-data-shootout/test/*'))
        gameS = []

        for name in gameS_long:
            want = name.split('/')[-1].split('.')[0]
            gameS.append(want)


        gameID_01_list = gameS   #所有测试视频的合集
    else:
        gameID_01_list = [ '3c993bd2_0', '3c993bd2_1' ]

    for gameID_01 in gameID_01_list:
        # import pudb; pu.db
        prob = np.load( f'../work/probS_npy/{SPLIT}/{gameID_01}.npy')
        frames_num = prob.shape[0]
        for i in range(frames_num):
            nameS__vdo_img.append(  f'{gameID_01}-{i+1:06}.jpg'  )

        probS_all_vdo.append(prob)


    probS_all_vdo = np.concatenate(probS_all_vdo, axis=0)


def submit(probS__all_vdo, nameS__vdo_img, roll_win = 2, nms_len_halfS = [16, 35, 16]):
    fps       = 25

    class2id = {"play":0, "throwin":1, "challenge":2}

    logits_df             = pd.DataFrame(probS__all_vdo, columns=['background','play', 'throwin', 'challenge'] )
    # logits_df             = pd.DataFrame(probS__all_vdo, columns=['background','challenge','play','throwin'] )
    logits_df['vdoName']  = nameS__vdo_img
    strS                  = logits_df['vdoName'].str.split('-').str
    logits_df['video_id'] = strS[0]
    logits_df['_frmID']   = strS[1].str.split('.').str[0].astype(int)
                                    # str.split('.').str[0]: 去掉.jpg

    DF_3events = pd.DataFrame()  # 存所有视频 所有event 的信息
    for video_id, df1vdo in logits_df.groupby('video_id'):  # 逐视频处理?
        for a_event in ['challenge', 'throwin', 'play']:  #  3者的顺序随意? 3个event独立
            df1vdo['当前处理的event__各event互补干扰'] = a_event  # 这行为了debug时方便看

            # pandas supports 4 types of windowing operations:
                    # Rolling window: Generic fixed or variable sliding window over the values.
                    #
                    # Weighted window: Weighted, non-rectangular window supplied by the scipy.signal library.
                    #
                    # Expanding window: Accumulating window over the values.
                    #
                    # Exponentially Weighted window: Accumulating and exponentially weighted window over the values.

            # large values may have an impact on windows, which do not include these values.
            # A: array
            A_logit = df1vdo[a_event].rolling(window=roll_win, center=True).mean().fillna(-4444).values
                                       # Moving averages are used to smooth out the data.
                                                                                   #        -4444 : 一个很负的数, 降序排列时 垫底就行
            df1vdo['rolling_logits'] = A_logit  #  # 这行是我自己加的吧 为了理解
            topIdx                 = np.argsort( - A_logit)  # 降序
            A_top                  = np.empty_like(topIdx)
            A_top[topIdx]          = np.arange( len(topIdx) )
            # index list for detected action
            idx_list = []
            for i in range(len(A_logit)):
                maxScore_id = topIdx[i]
                if maxScore_id != -444:  # 4:死掉了
                    # Add maximam index to index_list
                    idx_list.append(maxScore_id)  # 最终要的就是idx_list
                    for left_right in [-1,  1]:

                        if  a_event == 'play':
                            nms_len_half = nms_len_halfS[0]
                        elif  a_event == 'throwin':
                            nms_len_half = nms_len_halfS[1]
                        else:
                            nms_len_half = nms_len_halfS[2]

                        for step in range(1, nms_len_half+1 ):
                            supress_id = maxScore_id + step * left_right
                            if supress_id >= 0            and \
                              supress_id < len(A_logit):
                                topIdx[A_top[supress_id]] = -444
                                # Exclude frames near maxScore_id  (认为the action occurred at maxScore_id.)
                else:
                    # print('maxScore_id 小于0!',  maxScore_id)
                    pass
            # nms后, 89k多个被认为是事件的帧, 只剩6k多.
            df_1event_1vdo          = df1vdo \
                                         .iloc[idx_list] \
                                         .reset_index(drop=True) \
                                         .reset_index() \
                                         .rename(columns={'index':'conf_rank'}) [ ['conf_rank','video_id','_frmID'] ]
            df_1event_1vdo['event'] = a_event
            DF_3events        = DF_3events.append(df_1event_1vdo)
            # 有些帧 被同时认为是play的同时 还被认为是challenge 或throwin? 3个event独立

    DF_3events['time']  = DF_3events['_frmID'] / fps
    DF_3events['score'] = 1 / ( DF_3events['conf_rank']+1 )

    return DF_3events[  ['video_id', 'time', 'event', 'score']   ]


if  '修改自https://www.kaggle.com/code/ryanholbrook/competition-metric-dfl-event-detection-ap':
    gt = pd.read_csv(f"{P_data}/train.csv",
                     usecols=['video_id', 'time', 'event'],
                    )

    events2tol = {
        "challenge" :  [0.3  , 0.4  , 0.5  , 0.6  , 0.7]  ,
        "play"      :  [0.15 , 0.20 , 0.25 , 0.30 , 0.35] ,
        "throwin"   :  [0.15 , 0.20 , 0.25 , 0.30 , 0.35] ,
    }

    def filter_dets(det       :  pd.DataFrame ,
                        intervals :  pd.DataFrame ,
                        ) -> pd.DataFrame:
        """Drop detections not inside a scoring interval"""
        det_time = det.loc[:, 'time'].sort_values().to_numpy()
        intervals      = intervals.to_numpy()
        is_scored      = np.full_like(det_time, False, dtype=bool)

        i, j = 0, 0
        while i < len(det_time) and j < len(intervals):
            time = det_time[i]
            int_ = intervals[j]

            # If the detection is prior in time to the interval, go to the next detection.
            if time < int_.left:
                i += 1
            # If the detection is inside the interval, keep it and go to the next detection.
            elif time in int_:
                is_scored[i] = True
                i += 1
            # If the detection is later in time, go to the next interval.
            else:
                j += 1

        return det.loc[is_scored].reset_index(drop=True)


    def assign_gt2det(tolerance : float        ,
                    gt3event    : pd.DataFrame ,
                    dets        : pd.DataFrame ,
                   ) -> pd.DataFrame:
        # Match detections to ground truth events.
        # Arguments are taken from a common group::
                # event x tolerance x video evaluation
        dets_sorted = dets.sort_values('score', ascending=False).dropna()

        matched01 = np.full_like(dets_sorted['event'], False, dtype=bool)
        gt_matched_S = set()
        for i, a_det in enumerate(dets_sorted.itertuples(index=False)):
                                            # itertuples(index=False) : 逐行取表格
            min_error    = tolerance
            most_near_gt = None

            for a_gt in gt3event.itertuples(index=False):
                      # gt3event已经按时间排序
                error = abs(a_det.time - a_gt.time)
                if error < min_error and \
                  not a_gt in gt_matched_S:
                    most_near_gt = a_gt
                    min_error    = error

            if most_near_gt is not None:
                matched01[i] = True
                gt_matched_S.add(most_near_gt)

        dets_sorted['matched'] = matched01

        return dets_sorted


    from typing import  Tuple
    def P_R_curve(  matches: np.ndarray,
                  scores: np.ndarray,
                  TP_FN: int,
                 )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(matches) == 0:
            return [1], [0], []

        # Sort matches by decreasing confidence
        idxs    = np.argsort(scores, kind='stable')[::-1]
        scores  = scores[idxs]
        matches = matches[idxs]

        distinct_value_indices = np.where(np.diff(scores))[0]
        threshold_idxs         = np.r_[distinct_value_indices, matches.size - 1]
        thresholds             = scores[threshold_idxs]

        # Matches become TPs and
        # non-matches FPs as confidence threshold decreases
        TPs = np.cumsum(matches)[threshold_idxs]
        FPs = np.cumsum(~matches)[threshold_idxs]

        prec = TPs / (TPs + FPs)
        prec[np.isnan(prec)] = 0
        recall = TPs / TP_FN
                       # total number of ground truths,
                       # which might be different than total number of matches

        # Stop when full recall attained and reverse the outputs so recall is non-increasing.
        last_ind = TPs.searchsorted(TPs[-1])
        sl = slice(last_ind, None, -1)

        # Final precision is 1 and final recall is 0
        return np.r_[prec[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


    def get_ap(matches: np.ndarray, scores: np.ndarray, TP_FN: int) -> float:
        prec, recall, _ = P_R_curve(matches, scores, TP_FN)
        # Compute step integral
        return -np.sum(np.diff(recall) * np.array(prec)[:-1])


    from typing import Dict
    def event_ap(gt      : pd.DataFrame,
               yourCSV : pd.DataFrame,
               events2tol   : Dict[str, float],
               ) -> float:

        from pandas.testing import assert_index_equal
        assert_index_equal(gt.columns      , pd.Index(['video_id' , 'time' , 'event']))
        assert_index_equal(yourCSV.columns , pd.Index(['video_id' , 'time' , 'event' , 'score']))

        # Ensure gt and yourCSV are sorted properly
        gt      = gt.sort_values(     ['video_id' , 'time'])
        yourCSV = yourCSV.sort_values(['video_id' , 'time'])

        # Extract scoring intervals.
        intervals = (
                    gt
                    .query( "event in ['start', 'end']" )
                    .assign( interval =  lambda x: x.groupby(['video_id', 'event']).cumcount()  )
                    .pivot( index='interval', columns=['video_id', 'event'], values='time')
                    .stack('video_id')
                    .swaplevel()
                    .sort_index()
                    .loc[:, ['start', 'end']]
                    .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
                    )

        # Extract ground-truth events.
        gt3event = (
                    gt
                    .query("event not in ['start', 'end']")
                    .reset_index(drop=True)
                   )

        event_cnts = gt3event.value_counts('event').to_dict()
        # needed for recall calculation

        # Create table for detections with a column indicating a match to a ground-truth event
        dets = yourCSV.assign(matched = False)

        # Remove detections outside of scoring intervals
        detS_at_interval = []
        for (det_group, dets), (int_group, ints) in zip( dets.groupby('video_id'),
                                                        intervals.groupby('video_id'),
                                                        ):
            assert det_group == int_group
            detS_at_interval.append(filter_dets(dets, ints))
        detS_at_interval = pd.concat(detS_at_interval, ignore_index=True)

        # Create table of
            # event-class x tolerance x video_id values
            # ev            tol         vid 首字母ETV
        table_ETV = pd.DataFrame(
            [(ev, tol, vid)
                for ev in events2tol.keys()
                    for tol in events2tol[ev]
                        for vid in gt3event['video_id'].unique()
            ],
            columns=['event', 'tolerance', 'video_id'],
        )

        # Create match evaluation groups:
            # event-class x tolerance x video_id
        dets_grouped = (
            table_ETV
            .merge(detS_at_interval, on=['event', 'video_id'], how='left')
            .groupby(['event', 'tolerance', 'video_id'])
        )

        gt3event_grouped = (
            table_ETV
            .merge(gt3event, on=['event', 'video_id'], how='left')
            .groupby(['event', 'tolerance', 'video_id'])
        )

        dets_after_assign = []
        for key in table_ETV.itertuples(index=False):
            #  在各个ETV(evaluation group)里, assign  ground truth events to det
            dets = dets_grouped.get_group(key)
            gts  = gt3event_grouped.get_group(key)

            dets_after_assign.append(  assign_gt2det( dets['tolerance'].iloc[0] ,
                                       gts                       ,
                                       dets                      ,
                                     )
                              )

        dets_after_assign = pd.concat(dets_after_assign)

        # Compute AP
            #  per event x tolerance group
        event_names = gt3event['event'].unique()
        ap_table = (
            # 这有点多余: .query("event in @event_names")
            dets_after_assign
            .query("event in @event_names")
            .groupby(['event', 'tolerance']).apply(
                                                  lambda  group: get_ap(group['matched'].to_numpy(),
                                                                        group['score'].to_numpy(),
                                                                        event_cnts[group['event'].iat[0]],
                                                                       )
                                                                                     # group['event'].iat[0]  获得当前event名
                                                  )
        )

        mean_ap = ap_table.groupby('event').mean().mean()
                                            # Average over tolerance (不是加权平均, 只是简单平均)
                                                # then over event classes
        return mean_ap


if 0:
# if '搜 后处理参数':
    from loguru import logger
    log_file =  './my.log'
    os.system('mv -f {}  {}_bk'.format(log_file, log_file))
    logger.add(log_file)
    logger.info( '调参开始' )

    from itertools import product
    top_ap      = 0
    top_paraS   = []
    ap_roll_nms = []

    roll_win = 1
    from numpy.random import randint as randI

    # logger.info( '(ap_score, roll_win_lenth, nms_len_halfS)\n ' )

    for roll_win in range(1,15,1):
        logger.info(f'{roll_win=}')
        logger.info(f'      AP                   | nms_len_halfS ')
        trial    = 1
        while trial < 199:
    # for nms_len_halfS in list(product( range(13,20), range(29, 40), range(13,20)   ) ):
    # for nms_len_halfS in list(product( range(13,20), range(29, 40), range(13,20)   ) ):
            nms_len_halfS = [randI(1,50), randI(1, 75), randI(1,50)]
        #
            out_df = submit(probS_all_vdo, nameS__vdo_img, roll_win, nms_len_halfS)

            # import pudb; pu.db
            ap_score = event_ap( gt[   gt['video_id'].isin( out_df['video_id'].unique() ) ] ,
                                 out_df                                                     ,
                                 events2tol                                                      ,
                              )


            logger.info(f'      {100 * ap_score:.2f}, {nms_len_halfS}')

            if top_ap <= ap_score :
                top_ap = ap_score
                top_paraS.append( (top_ap, roll_win, nms_len_halfS) )

                logger.info(f'          目前最佳ap: {100 * ap_score:.2f}, nms: {nms_len_halfS}')



            trial += 1
            logger.info(f'          {trial=}')

    logger.info(f'top_ap是   {100 * ap_score:.2f}' )
    logger.info(f'{top_paraS= }')

out_df = submit(probS_all_vdo, nameS__vdo_img, roll_win, nms_len_halfS)

ap_score = event_ap( gt[   gt['video_id'].isin( out_df['video_id'].unique() ) ] ,
                     out_df                                                     ,
                     events2tol                                                      ,
                    )

