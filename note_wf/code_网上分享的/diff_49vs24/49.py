import sys
sys.path.append('../work/timm/pytorch-image-models')


import glob
import os

import cv2
import numpy as np
import torch

from tqdm import tqdm

from timm.data import ImageDataset, create_loader
torch.backends.cudnn.benchmark = True


VALID = False # for validation
TEST = True # for submission


# # Extract images


def extract_images(video_path, out_dir):
    video_name = os.path.basename(video_path).split('.')[0]
    cam = cv2.VideoCapture(video_path)
    print(video_path)
    frame_count = 1
    while True:
        successed, img = cam.read()
        if not successed:
            break
        outfile = f'{out_dir}/{video_name}-{frame_count:06}.jpg'
        img = cv2.resize(img,
                         dsize=IMG_SIZE,
                         interpolation=cv2.INTER_AREA,
                        )
                        # resampling using pixel area relation.
                        # It may be a preferred method for image decimation,
                        # (每十人杀一；降采样)
                        # as it gives moire'-free results.
                        # But when the image is zoomed,
                        # it is similar to the INTER_NEAREST method

                        # 默认:INTER_LINEAR
                        # bilinear interpolation
        cv2.imwrite(outfile, img)
        frame_count += 1

# IMG_SIZE = (456, 456)
IMG_SIZE = (512, 512)

if TEST:
    OUT_DIR = '/temp/work/extracted_images_test'
    IN_DIR = '../input/dfl-bundesliga-data-shootout/test'
    IN_VIDEOS = sorted(glob.glob('../input/dfl-bundesliga-data-shootout/test/*'))

if VALID:
    OUT_DIR = '/temp/work/extracted_images_train'
    IN_DIR = '../input/dfl-bundesliga-data-shootout/train'
    IN_VIDEOS = ['../input/dfl-bundesliga-data-shootout/train/3c993bd2_0.mp4','../input/dfl-bundesliga-data-shootout/train/3c993bd2_1.mp4']

# !mkdir -p $OUT_DIR
for video_path in IN_VIDEOS:
    extract_images(video_path, OUT_DIR)


# Classify images with timm loader.

class args:
    batch_size    = 32
    workers       = 2
    data          = OUT_DIR
    img_size      = None
    input_size    = [3, 512 , 512]
    interpolation = "box"
    mean          = None
    std           = None

@torch.no_grad()
def inference(args):
    # create model
    model = torch.jit.load("../input/aindao-dfl-baseline-v0/finetuned_model.pt", "cuda").eval()

    loader = create_loader(
        ImageDataset(args.data)                                  ,
        input_size     = args.input_size                         ,
        batch_size     = args.batch_size                         ,
        use_prefetcher = True                                    ,
        interpolation  = args.interpolation                      ,
        mean           = model.input_mean.cpu().numpy().tolist() ,
        std            = model.input_std.cpu().numpy().tolist()  ,
        num_workers    = args.workers                            ,
        crop_pct=1.0)

    prob = []
    for batch_idx, (input, _) in enumerate(loader):
        input = input.to("cuda", non_blocking=True)
        labels = model(input)
        prob.append(torch.softmax(labels, -1))

    prob = torch.cat(prob).cpu().numpy()
    return prob, loader.dataset.filenames(basename=True)


if VALID:
    prob_train, filenames_train = inference(args)
    np.savez("prob_train.npy", prob_train, filenames_train)


if TEST:
    prob_test, filenames_test = inference(args)
    np.savez("prob_test.npy", prob_test, filenames_test)


# # Extract frames where action is likely to have occurred using the Image Prediction Score.


err_tol = {
    'challenge': [ 0.30, 0.40, 0.50, 0.60, 0.70 ],
    'play': [ 0.15, 0.20, 0.25, 0.30, 0.35 ],
    'throwin': [ 0.15, 0.20, 0.25, 0.30, 0.35 ]
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
event_names = ['challenge', 'throwin', 'play']
label_dict = {
    'background':0,
    'challenge':1,
    'play':2,
    'throwin':3,
}
event_names_with_background = ['background','challenge','play','throwin']


def make_sub(prob, filenames):

    frame_rate   = 25
    window_size  = 2
    ignore_width = 16

    df = pd.DataFrame(prob,columns=event_names_with_background)
    df['video_name'] = filenames
    df['video_id'] = df['video_name'].str.split('-').str[0]
    df['frame_id'] = df['video_name'].str.split('-').str[1].str.split('.').str[0].astype(int)

    train_df = []
    for video_id,gdf in df.groupby('video_id'):
        for i, event in enumerate(event_names):
            #print(video_id, event)
            prob_arr = gdf[event].rolling(window=window_size, center=True).mean().fillna(-100).values
            sort_arr = np.argsort(-prob_arr)
            rank_arr = np.empty_like(sort_arr)
            rank_arr[sort_arr] = np.arange(len(sort_arr))
            idx_list = []
            for i in range(len(prob_arr)):
                this_idx = sort_arr[i]
                if this_idx >= 0:
                    idx_list.append(this_idx)
                    for parity in (-1,1):
                        for j in range(1, ignore_width+1):
                            ex_idx = this_idx + j * parity
                            if ex_idx >= 0 and ex_idx < len(prob_arr):
                                sort_arr[rank_arr[ex_idx]] = -1
            this_df = gdf.reset_index(drop=True).iloc[idx_list].reset_index(drop=True)
            this_df["score"] = prob_arr[idx_list]
            this_df['event'] = event
            train_df.append(this_df)
    train_df = pd.concat(train_df)
    train_df['time'] = train_df['frame_id']/frame_rate

    return train_df.reset_index(drop=True)


# copy from https://www.kaggle.com/code/ryanholbrook/competition-metric-dfl-event-detection-ap

import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from typing import Dict, Tuple

tolerances = {
    "challenge": [0.3, 0.4, 0.5, 0.6, 0.7],
    "play": [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin": [0.15, 0.20, 0.25, 0.30, 0.35],
}


def filter_detections(
        detections: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    """Drop detections not inside a scoring interval."""
    detection_time = detections.loc[:, 'time'].sort_values().to_numpy()
    intervals = intervals.to_numpy()
    is_scored = np.full_like(detection_time, False, dtype=bool)

    i, j = 0, 0
    while i < len(detection_time) and j < len(intervals):
        time = detection_time[i]
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

    return detections.loc[is_scored].reset_index(drop=True)


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


def event_detection_ap(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        tolerances: Dict[str, float],
) -> float:

    assert_index_equal(solution.columns, pd.Index(['video_id', 'time', 'event']))
    assert_index_equal(submission.columns, pd.Index(['video_id', 'time', 'event', 'score']))

    # Ensure solution and submission are sorted properly
    solution = solution.sort_values(['video_id', 'time'])
    submission = submission.sort_values(['video_id', 'time'])

    # Extract scoring intervals.
    intervals = (
        solution
        .query("event in ['start', 'end']")
        .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
        .pivot(index='interval', columns=['video_id', 'event'], values='time')
        .stack('video_id')
        .swaplevel()
        .sort_index()
        .loc[:, ['start', 'end']]
        .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
    )

    # Extract ground-truth events.
    ground_truths = (
        solution
        .query("event not in ['start', 'end']")
        .reset_index(drop=True)
    )

    # Map each event class to its prevalence (needed for recall calculation)
    class_counts = ground_truths.value_counts('event').to_dict()

    # Create table for detections with a column indicating a match to a ground-truth event
    detections = submission.assign(matched = False)

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

solution = pd.read_csv("../input/dfl-bundesliga-data-shootout/train.csv", usecols=['video_id', 'time', 'event'])


if VALID:
    train_df = make_sub(prob_train, filenames_train)
    score = event_detection_ap(solution[solution['video_id'].isin(train_df['video_id'].unique())], train_df[['video_id', 'time', 'event', 'score']], tolerances)
    print(score)


if TEST:
    test_df = make_sub(prob_test, filenames_test)
    test_df[['video_id', 'time', 'event', 'score']].to_csv("submission.csv", index=False)


(train_df if VALID else test_df)[['video_id', 'time', 'event', 'score']]


#
