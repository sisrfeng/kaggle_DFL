# [Competition Metric - DFL Event Detection AP]
# https://www.kaggle.com/code/ryanholbrook/competition-metric-dfl-event-detection-ap
#######################
import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from typing import Dict, Tuple


# 用于设置函数中使用的 best_error 的初始值
tolerances = {
    "challenge" : [0.3, 0.4, 0.5, 0.6, 0.7],
    "play"      : [0.15, 0.20, 0.25, 0.30, 0.35],
    "throwin"   : [0.15, 0.20, 0.25, 0.30, 0.35],
}


def filter_detections(detections: pd.DataFrame,
                    intervals: pd.DataFrame,
                     ) -> pd.DataFrame:
    # Drop detections not inside a scoring interval.
    detection_time = detections.loc[:, 'time'].sort_values().to_numpy()
    intervals = intervals.to_numpy()
    # np.full_like: detection_timeと同じshapeでFalseを埋める
    is_scored = np.full_like(detection_time, False, dtype=bool)

    i, j = 0, 0
    # while文の中では、detection_timeが startとendの間にあると、False->Trueに変換
    while i < len(detection_time) and j < len(intervals):
        time = detection_time[i]
        int_ = intervals[j]

        # If the detection is prior in time to the interval, go to the next detection.
        # int_.leftはstartの時刻を示す。それよりもtimeが小さいならば、iをインクリメント
        if time < int_.left:
            i += 1
        # If the detection is inside the interval, keep it and go to the next detection.
        # int_にtimeが含まれていればis_scored[i]をFalse->Trueに変える
        elif time in int_:
            is_scored[i] = True
            i += 1
        # If the detection is later in time, go to the next interval.
        # time > int_.rightならばjをインクリメント
        else:
            j += 1

    return detections.loc[is_scored].reset_index(drop=True)


def match_detections(
    tolerance: float,
    ground_truths: pd.DataFrame,
    detections: pd.DataFrame
) -> pd.DataFrame:
    """Match detections to ground truth events.
    Arguments are taken from a common event x tolerance x video evaluation group.
    detectionsとground truth(正解)を照合する。
    eventとtolerance毎に照合を行う。
    """
    # Scoreでソート
    detections_sorted = detections.sort_values('score', ascending=False).dropna()
    # is_matchedというdetections_sorted['event']と同じ行数のFalseだけのarrayを用意
    is_matched = np.full_like(detections_sorted['event'], False, dtype=bool)

    # 繰り返し処理
    gts_matched = set()
    for i, det in enumerate(detections_sorted.itertuples(index=False)):
        best_error = tolerance # best_errorの初期値設定
        best_gt = None

        for gt in ground_truths.itertuples(index=False):
            error = abs(det.time - gt.time) # timeのズレ分をerrorとする
            # errorがbest かつ　gtが新規ならば更新
            if error < best_error and not gt in gts_matched:
                best_gt = gt
                best_error = error

        if best_gt is not None:
            is_matched[i] = True
            gts_matched.add(best_gt)

    # is_matchedを加える
    detections_sorted['matched'] = is_matched

    return detections_sorted


def precision_recall_curve(
    matches: np.ndarray,
    scores: np.ndarray,
    p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """matches. scores, pからprecision、recallを求める
    """
    if len(matches) == 0:
        return [1], [0], []

    # Sort matches by decreasing confidence
    idxs = np.argsort(scores, kind='stable')[::-1]
    scores = scores[idxs]
    matches = matches[idxs]

    distinct_value_indices = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, matches.size - 1] # np.r_でdistinct_value_indicesとmatches.size - 1を結合
    thresholds = scores[threshold_idxs]

    # Matches become TPs and non-matches FPs as confidence threshold decreases
    tps = np.cumsum(matches)[threshold_idxs]  # Trueの総和
    fps = np.cumsum(~matches)[threshold_idxs] # Falseの総和

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
    """メインのKPI
    """

    assert_index_equal(solution.columns, pd.Index(['video_id', 'time', 'event']))
    assert_index_equal(submission.columns, pd.Index(['video_id', 'time', 'event', 'score']))

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
            match_detections(dets['tolerance'].iloc[0], gts, dets) # ここでmatch_detections関数を実行
        )
    detections_matched = pd.concat(detections_matched)

    # Compute AP per event x tolerance group
    event_classes = ground_truths['event'].unique()
    ap_table = (
        detections_matched
        .query("event in @event_classes")
        .groupby(['event', 'tolerance']).apply(
        lambda group: average_precision_score(   # ここでaverage_precision_score関数を実行
        group['matched'].to_numpy(),
                group['score'].to_numpy(),
                class_counts[group['event'].iat[0]],
            )
        )
    )

    # Average over tolerances, then over event classes
    mean_ap = ap_table.groupby('event').mean().mean()

    return mean_ap


# # Example #


# Let's walk through a few examples, using the training set labels as a stand-in for the ground-truth labels.


from pathlib import Path

data_dir = Path('../input/dfl-bundesliga-data-shootout')


solution = pd.read_csv(data_dir / 'train.csv', usecols=['video_id', 'time', 'event'])
solution.head()


# The submission should be the `video_id`, `time`, and `event` columns, dropping the `start` and `end` events which delimit the scoring intervals.


perfect_submission = (
    solution
    .query("event not in ['start', 'end']")
    .reset_index(drop=True)
    .assign(score = 1.0)
)
perfect_submission.head()


# Running this through our metric, we see indeed that this results in a perfect score.


event_detection_ap(solution, perfect_submission, tolerances)


# Now let's see what happens if we shuffle about 10% of the event labels.


noisy_submission = perfect_submission.copy()
idx = noisy_submission.sample(frac=0.1).index
noisy_submission.loc[idx, 'event'] = noisy_submission.loc[idx, 'event'].sort_index().to_numpy()

event_detection_ap(solution, noisy_submission, tolerances)


# And finally we'll try shuffling the timestamps.


noisy_submission_2 = perfect_submission.copy()
time_noise = np.random.normal(loc=0.0, scale=0.15, size=noisy_submission.shape[0])
noisy_submission_2['time'] = noisy_submission_2['time'] + time_noise

event_detection_ap(solution, noisy_submission_2, tolerances)


# ---


# <a id='#2'></a>
# <h1 align='left'>関数の中身理解 / Understanding the content of functions </h1>


# ## MAIN： event_detection_ap(solution, noisy_submission, tolerances)
# この関数がメインなので、これを上から順番に確認していきます。
# Since this function is the main, we will check it in order from the top.


# solution
print(perfect_submission.shape)
perfect_submission.head(3)


# noisy_submission
noisy_submission = perfect_submission.copy()
idx = noisy_submission.sample(frac=0.1).index # 10%だけidxを選ぶ
idx


# noisy_submissionの'event'列をシャッフルする。
noisy_submission.loc[idx, 'event'] = noisy_submission.loc[idx, 'event'].sort_index().to_numpy()


solution


noisy_submission


# indexが'video_id', 'time', 'event'を含むかを確認
assert_index_equal(solution.columns, pd.Index(['video_id', 'time', 'event']))
# AssertionError: Index are different

# エラー発生時は下記のように出力
    # Index values are different (33.33333 %)
    # [left]:  Index(['video_id', 'time', 'event'], dtype='object')
    # [right]: Index(['video_id', 'time', 'event__'], dtype='object')


# 問題ない場合は出力は何もない
submission = noisy_submission
assert_index_equal(submission.columns, pd.Index(['video_id', 'time', 'event', 'score']))


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

intervals


# startとendのみを切り出し
intervals_ = (
    solution
    .query("event in ['start', 'end']")
#     .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
#     .pivot(index='interval', columns=['video_id', 'event'], values='time')
#     .stack('video_id')
#     .swaplevel()
#     .sort_index()
#     .loc[:, ['start', 'end']]
#     .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
)

intervals_


# cumcountで累積回数をナンバリング
intervals_ = (
    solution
    .query("event in ['start', 'end']")
    .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
#     .pivot(index='interval', columns=['video_id', 'event'], values='time')
#     .stack('video_id')
#     .swaplevel()
#     .sort_index()
#     .loc[:, ['start', 'end']]
#     .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
)

intervals_


# index='interval', columns=['video_id', 'event'], values='time'でinterval毎にピボット
intervals_ = (
    solution
    .query("event in ['start', 'end']")
    .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
    .pivot(index='interval', columns=['video_id', 'event'], values='time')
#     .stack('video_id')
#     .swaplevel()
#     .sort_index()
#     .loc[:, ['start', 'end']]
#     .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
)

intervals_


# video_id毎にスタックする。
intervals_ = (
    solution
    .query("event in ['start', 'end']")
    .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
    .pivot(index='interval', columns=['video_id', 'event'], values='time')
    .stack('video_id')
#     .swaplevel()
#     .sort_index()
#     .loc[:, ['start', 'end']]
#     .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
)

intervals_


# video_idとintervalをswap
intervals_ = (
    solution
    .query("event in ['start', 'end']")
    .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
    .pivot(index='interval', columns=['video_id', 'event'], values='time')
    .stack('video_id')
    .swaplevel()
#     .sort_index()
#     .loc[:, ['start', 'end']]
#     .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
)

intervals_


# intervalでソート
intervals_ = (
    solution
    .query("event in ['start', 'end']")
    .assign(interval=lambda x: x.groupby(['video_id', 'event']).cumcount())
    .pivot(index='interval', columns=['video_id', 'event'], values='time')
    .stack('video_id')
    .swaplevel()
    .sort_index()
#     .loc[:, ['start', 'end']]
#     .apply(lambda x: pd.Interval(*x, closed='both'), axis=1)
)

intervals_


# pd.Intervalでスライス区切りのような形に変形
intervals_ = (
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

intervals_


# Extract ground-truth events. (event列でstart, end以外のものを抜き出す。)
ground_truths = (
    solution
    .query("event not in ['start', 'end']")
    .reset_index(drop=True)
)
ground_truths


# Map each event class to its prevalence (needed for recall calculation)
# eventとその回数の辞書をつくる
class_counts = ground_truths.value_counts('event').to_dict()
class_counts


# Create table for detections with a column indicating a match to a ground-truth event
# matchedという列をsubmissionに追加し、全てFalseを入れて、detectionsというDataFrameにすうｒ．
detections = submission.assign(matched = False)
detections


# Remove detections outside of scoring intervals
detections_filtered = []

for (det_group, dets), (int_group, ints) in zip(
    detections.groupby('video_id'), intervals.groupby('video_id')
):
    assert det_group == int_group
    detections_filtered.append(filter_detections(dets, ints))
detections_filtered = pd.concat(detections_filtered, ignore_index=True)

detections_filtered


# はじめの１データだけでforループを止めて中身を確認
for (det_group, dets), (int_group, ints) in zip(
    detections.groupby('video_id'), intervals.groupby('video_id')
):
    assert det_group == int_group
    detections_filtered.append(filter_detections(dets, ints))
    break


# detectionsから、det_group==video_idとdets->video_idのデータを呼び出す。
(det_group, dets)


# intervals、int_group==video_idとdets->video_idのデータを呼び出す。
(int_group, ints)


# 互いのvideo_idが一致しているか確認
assert det_group == int_group


# filter_detections関数を実行
# detections_filtered.append(filter_detections(dets, ints))
filter_detections(dets, ints)


# ---


# ## filter_detections関数の中身確認
# ここで、filter_detections関数を実行しています。その中身を確認します。
# Here I am running the filter_detections function. Check its contents.


# detectionsのtimeをnp.arrayとしてdetection_timeに渡す
detections_ = dets
intervals_  = ints
detection_time = detections_.loc[:, 'time'].sort_values().to_numpy()
print(len(detection_time))
detection_time[:4]


# numpy化
intervals_ = intervals_.to_numpy()
intervals_[:4]


# np.full_like: detection_timeと同じshapeでFalseを埋める
is_scored = np.full_like(detection_time, False, dtype=bool)
print(len(is_scored))
is_scored[:4]


# while文の中がどのように処理されるのか確認するための関数
def print_ij(i, j):
    if i < 10 and j < 10:
        print(
            f"i: {i}, j: {j},\n\
            detection_time[i]:{detection_time[i]},\n\
            intervals_[j]: {intervals_[j]}\n\
            is_scored[i]:{is_scored[i]} "
        )


# while文の中では、detection_timeが startとendの間にあると、False->Trueに変換
i, j = 0, 0
while i < len(detection_time) and j < len(intervals_):
    time = detection_time[i]
    int_ = intervals_[j]

    # If the detection is prior in time to the interval, go to the next detection.
    # int_.leftはstartの時刻を示す。それよりもtimeが小さいならば、iをインクリメント
    if time < int_.left:
        i += 1
        print_ij(i-1, j)
    # If the detection is inside the interval, keep it and go to the next detection.
    # int_にtimeが含まれていればis_scored[i]をFalse->Trueに変える
    elif time in int_:
        is_scored[i] = True
        i += 1
        print_ij(i-1, j)
    # If the detection is later in time, go to the next interval.
    # time > int_.rightならばjをインクリメント
    else:
        j += 1
        print_ij(i, j-1)

is_scored[:4]


# return detections.loc[is_scored].reset_index(drop=True)
detections_.loc[is_scored].reset_index(drop=True)


# ### ↑  filter_detections関数の中身確認完了


# ---


# Create table of event-class x tolerance x video_id values
aggregation_keys = pd.DataFrame(
    [(ev, tol, vid)
     for ev in tolerances.keys()
     for tol in tolerances[ev]
     for vid in ground_truths['video_id'].unique()],
    columns=['event', 'tolerance', 'video_id'],
)

aggregation_keys


tolerances


ground_truths['video_id'].unique()


# detections_filtered やground_truthsをaggregation_keysにマージする。
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


detections_grouped_ = (
    aggregation_keys
    .merge(detections_filtered, on=['event', 'video_id'], how='left')
#     .groupby(['event', 'tolerance', 'video_id'])
)

detections_grouped_


ground_truths_grouped_ = (
    aggregation_keys
    .merge(ground_truths, on=['event', 'video_id'], how='left')
#     .groupby(['event', 'tolerance', 'video_id'])
)
ground_truths_grouped_


# Match detections to ground truth events by evaluation group
detections_matched = []
for key in aggregation_keys.itertuples(index=False):
    dets = detections_grouped.get_group(key)
    gts = ground_truths_grouped.get_group(key)
    detections_matched.append(
        match_detections(dets['tolerance'].iloc[0], gts, dets)
    )
detections_matched = pd.concat(detections_matched)

detections_matched


# Match detections to ground truth events by evaluation group
detections_matched_ = []
for key in aggregation_keys.itertuples(index=False):
    dets = detections_grouped.get_group(key)
    gts = ground_truths_grouped.get_group(key)
    # ここでmatch_detections関数を使います。
    detections_matched_.append(
        match_detections(dets['tolerance'].iloc[0], gts, dets)
    )
    break


dets.head()


gts.head()


match_detections(dets['tolerance'].iloc[0], gts, dets)


# ---
# ## match_detections関数の確認
# ここで、match_detections関数を実行しています。その中身を確認します。
# Here I am running the match_detections function. Check its contents.


# def match_detections(
#     tolerance: float,
#     ground_truths: pd.DataFrame,
#     detections: pd.DataFrame
# ) -> pd.DataFrame:
tolerance_, ground_truths_, detections_=dets['tolerance'].iloc[0], gts, dets


# Scoreでソート
detections_sorted = detections_.sort_values('score', ascending=False).dropna()
print(detections_sorted.shape)
detections_sorted.head()


# is_matchedというFalseのarrayを作る
is_matched = np.full_like(detections_sorted['event'], False, dtype=bool)
print(len(is_matched))
is_matched


tolerance_


gts_matched = set()

for i, det in enumerate(detections_sorted.itertuples(index=False)):
    if i < 5:
        print(i)
        print(f"det: {det}")
    best_error = tolerance_ # best_errorの初期値設定
    best_gt = None

    j = 0
    for gt in ground_truths.itertuples(index=False):
        error = abs(det.time - gt.time) # timeのズレ分をerrorとする

        if i<5 and j < 5:
            print(f"gt: {gt}")
            print(f"error: {error}, best_error: {best_error}")
            j += 1

        # errorがbest かつ　gtが新規ならば更新
        if error < best_error and not gt in gts_matched:
            best_gt = gt
            best_error = error

    if best_gt is not None:
        is_matched[i] = True
        gts_matched.add(best_gt)


gts_matched


# detections_sorted['matched'] = is_matched
# return detections_sorted
detections_sorted['matched'] = is_matched
detections_sorted.head(3)


# ### ↑ ここまででmatch_detections関数の中身確認完了


# ---


# Compute AP per event x tolerance group
event_classes = ground_truths['event'].unique()
ap_table = (
    detections_matched
    .query("event in @event_classes")
    .groupby(['event', 'tolerance']).apply(
    lambda group: average_precision_score(   # average_precision_score関数をここで使っている。
    group['matched'].to_numpy(),
            group['score'].to_numpy(),
            class_counts[group['event'].iat[0]],
        )
    )
)

ap_table


# ---
# ## average_precision_score関数の確認
# ここで、average_precision_score関数を実行しています。その中身を確認します。
# Here I am running the average_precision_score function. Check its contents.


ap_table_ = (
    detections_matched
    .query("event in @event_classes")
#     .groupby(['event', 'tolerance']) #.apply(
#     lambda group: average_precision_score(   # average_precision_score関数をここで使っている。
#     group['matched'].to_numpy(),
#             group['score'].to_numpy(),
#             class_counts[group['event'].iat[0]],
#         )
#     )
)
ap_table_


# group['matched'].to_numpy()
matches = ap_table_.query('event=="challenge" & tolerance==0.30')['matched'].to_numpy()
print(matches.shape)
matches[:3]


# group['score'].to_numpy()
scores = ap_table_.query('event=="challenge" & tolerance==0.30')['score'].to_numpy()
print(scores.shape)
matches[:3]


# class_counts[group['event'].iat[0]]
print(class_counts)
tmp = ap_table_.query('event=="challenge"')["event"]
p = class_counts[tmp.iat[0]] # 0行目をiat[0]で取得 -> この例ではclass_counts["challenge"] となる
p


# def average_precision_score(matches: np.ndarray, scores: np.ndarray, p: int) -> float:
#     precision, recall, _ = precision_recall_curve(matches, scores, p)
#     # Compute step integral
#     return -np.sum(np.diff(recall) * np.array(precision)[:-1])

precision, recall, _ = precision_recall_curve(matches, scores, p) # precision_recall_curve関数を実行
print(precision, recall, _)
print(-np.sum(np.diff(recall) * np.array(precision)[:-1]))


# ---
# ## precision_recall_curve関数の確認
# ここで、precision_recall_curve関数を実行しています。その中身を確認します。
# Here I am running the precision_recall_curve function. Check its contents.


# def precision_recall_curve(
#     matches: np.ndarray,
#     scores: np.ndarray,
#     p: int
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

# matchesが何もない場合
# if len(matches) == 0:
#     return [1], [0], []


# Sort matches by decreasing confidence
idxs = np.argsort(scores, kind='stable')[::-1] # scoresをソート
scores_ = scores[idxs]
matches_ = matches[idxs]

print(idxs[:3])
print(scores_[:3])
print(matches_[:3])


distinct_value_indices = np.where(np.diff(scores))[0]
threshold_idxs = np.r_[distinct_value_indices, matches.size - 1] # np.r_でdistinct_value_indicesとmatches.size - 1を結合
thresholds = scores[threshold_idxs]

print(distinct_value_indices)
print(threshold_idxs)
print(thresholds)


# Matches become TPs and non-matches FPs as confidence threshold decreases
tps = np.cumsum(matches)[threshold_idxs]  # Trueの総和
fps = np.cumsum(~matches)[threshold_idxs] # Falseの総和

precision = tps / (tps + fps)
precision[np.isnan(precision)] = 0
recall = tps / p  # total number of ground truths might be different than total number of matches

print(f"p: {p}")
print(f"tps: {tps}, fps: {fps}")
print(f"precision: {precision}, recall: {recall}")


# Stop when full recall attained and reverse the outputs so recall is non-increasing.
last_ind = tps.searchsorted(tps[-1])
sl = slice(last_ind, None, -1)

print(f"last_ind: {last_ind}")
print(f"sl: {sl}")


# Final precision is 1 and final recall is 0
# return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]
np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


# ### ↑ precision_recall_curve関数の確認が完了
# ### ↑ average_precision_score関数の確認が完了


# ---


event_classes


ap_table


# Average over tolerances, then over event classes
mean_ap = ap_table.groupby('event').mean().mean()
# return mean_ap
mean_ap


# # Fin
