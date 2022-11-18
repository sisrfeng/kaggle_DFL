if "wf_package":
    print('注意kaggle kernel里的文件路径和读写权限!!!!!!!!!!!!!!!')
    # !mkdir ....

    import sys
    sys.path.append('../work/timm/pytorch-image-models')   # train.py里没有:

    import glob
    import os
    from tqdm.auto import tqdm
    from multiprocessing import Pool, cpu_count
    import cv2
    import time
    import argparse
    import logging
    import numpy as np
    import torch

    from timm.models import create_model, apply_test_time_pool
    from timm.data   import ImageDataset, create_loader, resolve_data_config
    from timm.utils  import AverageMeter, setup_default_logging

SPLIT =  'Val'
# SPLIT =  'Test'
IMG_SIZE = (456 , 456)

if 'Extract frames where action is likely to have occurred, by using the Image Prediction Score':

    err_tol = {
        'challenge' :  [ 0.30, 0.40, 0.50, 0.60, 0.70 ] ,
        'play'      :  [ 0.15, 0.20, 0.25, 0.30, 0.35 ] ,
        'throwin'   :  [ 0.15, 0.20, 0.25, 0.30, 0.35 ] ,
    }

    video_id_split = {
        'val':[
             '3c993bd2_0',
             '3c993bd2_1',
        ],
        'train':[
             '1606b0e6_0',
             '1606b0e6_1',
             '35bd9041_0',
             '35bd9041_1',
             '407c5a9e_1',
             '4ffd5986_0',
             '9a97dae4_1',
             'cfbe2e94_0',
             'cfbe2e94_1',
             'ecf251d4_0',
        ]
    }

    events3 = ['challenge', 'throwin', 'play']
    label_dict = {
        'background' :  0 ,
        'challenge'  :  1 ,
        'play'       :  2 ,
        'throwin'    :  3 ,
    }
    events4 = ['background','challenge','play','throwin']


    def Submit(prob, f_names):
        frame_rate   = 25
        window_size  = 10
        ignore_width = 10
        group_count  = 5

        df               = pd.DataFrame(prob, columns=events4 )
        df['video_name'] = f_names
        df['video_id']   = df['video_name'].str.split('-').str[0]
        df['frame_id']   = df['video_name'].str.split('-').str[1].str.split('.').str[0].astype(int)

        train_df = pd.DataFrame()
        for video_id,gdf in df.groupby('video_id'):
            for i, event in enumerate(events3):
                # Moving averages are used to smooth out the data.
                prob_arr = gdf[event].rolling(window=window_size, center=True).mean().fillna(-100).values
                gdf['rolling_prob'] = prob_arr
                sort_arr = np.argsort(-prob_arr)
                rank_arr = np.empty_like(sort_arr)
                rank_arr[sort_arr] = np.arange(len(sort_arr))
                # index list for detected action
                idx_list = []
                for i in range(len(prob_arr)):
                    this_idx = sort_arr[i]
                    if this_idx >= 0:
                        # Add maximam index to index_list
                        idx_list.append(this_idx)
                        for parity in (-1,1):
                            for j in range(1, ignore_width+1):
                                ex_idx = this_idx + j * parity
                                if ex_idx >= 0 and ex_idx < len(prob_arr):
                                    # Exclude frames near this_idx where the action occurred.
                                    sort_arr[rank_arr[ex_idx]] = -1
                this_df = gdf.iloc[idx_list].reset_index(drop=True).reset_index().rename(columns={'index':'rank'})[['rank','video_id','frame_id']]
                this_df['event'] = event
                train_df = train_df.append(this_df)

        train_df['time']  = train_df['frame_id'] / 25
        train_df['score'] = 1 / ( train_df['rank']+1 )

        return train_df

    if  'copy from https://www.kaggle.com/code/ryanholbrook/competition-metric-dfl-event-detection-ap':
        import numpy as np
        import pandas as pd
        from pandas.testing import assert_index_equal
        from typing import Dict, Tuple

        tolerances = {
            "challenge" :  [0.3  , 0.4  , 0.5  , 0.6  , 0.7]  ,
            "play"      :  [0.15 , 0.20 , 0.25 , 0.30 , 0.35] ,
            "throwin"   :  [0.15 , 0.20 , 0.25 , 0.30 , 0.35] ,
        }
        def filter_detections(det       :  pd.DataFrame ,
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


        def match_detections(
                tolerance: float, ground_truths: pd.DataFrame, detections: pd.DataFrame
        ) -> pd.DataFrame:
            """Match detections to ground truth events. Arguments are taken from a common event x tolerance x video evaluation group."""
            detections_sorted = detections.sort_values('score', ascending=False).dropna()

            is_matched = np.full_like(detections_sorted['event'], False, dtype=bool)
            gts_matched = set()
            for i, det in enumerate(detections_sorted.itertuples(index=False)):
                best_error = tolerance
                best_gt = None

                for gt in ground_truths.itertuples(index=False):
                    error = abs(det.time - gt.time)
                    if error < best_error and not gt in gts_matched:
                        best_gt = gt
                        best_error = error

                if best_gt is not None:
                    is_matched[i] = True
                    gts_matched.add(best_gt)

            detections_sorted['matched'] = is_matched

            return detections_sorted


        def precision_recall_curve(
                matches: np.ndarray, scores: np.ndarray, p: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if len(matches) == 0:
                return [1], [0], []

            # Sort matches by decreasing confidence
            idxs = np.argsort(scores, kind='stable')[::-1]
            scores = scores[idxs]
            matches = matches[idxs]

            distinct_value_indices = np.where(np.diff(scores))[0]
            threshold_idxs = np.r_[distinct_value_indices, matches.size - 1]
            thresholds = scores[threshold_idxs]

            # Matches become TPs and non-matches FPs as confidence threshold decreases
            tps = np.cumsum(matches)[threshold_idxs]
            fps = np.cumsum(~matches)[threshold_idxs]

            precision = tps / (tps + fps)
            precision[np.isnan(precision)] = 0
            recall = tps / p  # total number of ground truths might be different than total number of matches

            # Stop when full recall attained and reverse the outputs so recall is non-increasing.
            last_ind = tps.searchsorted(tps[-1])
            sl = slice(last_ind, None, -1)

            # Final precision is 1 and final recall is 0
            return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


        def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
            precision, recall, _ = precision_recall_curve(matches, scores, p)
            # Compute step integral
            return -np.sum(np.diff(recall) * np.array(precision)[:-1])


        def event_ap(gt : pd.DataFrame,
                               out_csv : pd.DataFrame,
                               tolerances : Dict[str, float],
                              ) -> float:

            assert_index_equal(gt.columns         , pd.Index(['video_id' , 'time' , 'event']))
            assert_index_equal(out_csv.columns , pd.Index(['video_id' , 'time' , 'event' , 'score']))

            # Ensure gt and out_csv are sorted properly
            gt      = gt.sort_values(['video_id'      , 'time'])
            out_csv = out_csv.sort_values(['video_id' , 'time'])

            # Extract scoring intervals.
                # import pudb; pu.db
                # 不用pudb就不会在pivot处报错
            intervals = (
                gt
                .query( "event in ['start', 'end']" )
                .assign( interval=lambda x: x.groupby(['video_id', 'event']).cumcount() )
                .pivot(index='interval', columns=['video_id', 'event'], values='time')
                .stack('video_id')
                .swaplevel()
                .sort_index()
                .loc[:, ['start', 'end']]
                .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
            )
            print('--------intervals')
            print(intervals)
            # Extract ground-truth events.
            ground_truths = (
                gt
                .query("event not in ['start', 'end']")
                .reset_index(drop=True)
            )
            print(f'---------------ground_truths')
            print(ground_truths)

            # map each event class to its prevalence (needed for recall calculation)
            class_counts = ground_truths.value_counts('event').to_dict()

            # Create table for detections with a column indicating a match to a ground-truth event
            detections = out_csv.assign(matched = False)

            # Remove detections outside of scoring intervals
            detections_filtered = []
            for (det_group, dets), (int_group, ints) in zip(
                detections.groupby('video_id'), intervals.groupby('video_id')
            ):
                assert det_group == int_group
                detections_filtered.append(filter_detections(dets, ints))
            detections_filtered = pd.concat(detections_filtered, ignore_index=True)

            # Create table of event-class x tolerance x video_id values
            aggregation_keys = pd.DataFrame(
                [(ev, tol, vid)
                 for ev in tolerances.keys()
                 for tol in tolerances[ev]
                 for vid in ground_truths['video_id'].unique()],
                columns=['event', 'tolerance', 'video_id'],
            )

            # Create match evaluation groups: event-class x tolerance x video_id
            detections_grouped = (
                aggregation_keys
                .merge(detections_filtered, on=['event', 'video_id'], how='left')
                .groupby(['event', 'tolerance', 'video_id'])
            )
            ground_truths_grouped = (
                aggregation_keys
                .merge(ground_truths, on=['event', 'video_id'], how='left')
                .groupby(['event', 'tolerance', 'video_id'])
            )

            # Match detections to ground truth events by evaluation group
            detections_matched = []
            for key in aggregation_keys.itertuples(index=False):
                dets = detections_grouped.get_group(key)
                gts = ground_truths_grouped.get_group(key)
                detections_matched.append(
                    match_detections(dets['tolerance'].iloc[0], gts, dets)
                )
            detections_matched = pd.concat(detections_matched)

            # Compute AP per event x tolerance group
            event_classes = ground_truths['event'].unique()
            ap_table = (
                detections_matched
                .query("event in @event_classes")
                .groupby(['event', 'tolerance']).apply(
                lambda group: average_precision_score(
                group['matched'].to_numpy(),
                        group['score'].to_numpy(),
                        class_counts[group['event'].iat[0]],
                    )
                )
            )

            # Average over tolerances, then over event classes
            mean_ap = ap_table.groupby('event').mean().mean()

            return mean_ap

        gt = pd.read_csv("../input/dfl-bundesliga-data-shootout/train.csv",
                               usecols=['video_id', 'time', 'event'],
                              )

out_df = pd.read_csv('~/.t/out_df.csv')

if SPLIT == 'Val':
    # 显示score
    print( event_ap( gt[ gt['video_id'].isin( out_df['video_id'].unique() ) ] ,
                     out_df[ ['video_id', 'time', 'event', 'score'] ]             ,
                     tolerances                                           ,
                              )
         )

else:
    out_df[ ['video_id', 'time', 'event', 'score'] ].to_csv("submit.csv", index=False)


