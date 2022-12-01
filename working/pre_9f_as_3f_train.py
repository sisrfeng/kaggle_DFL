#  [Training part](https://www.kaggle.com/code/kmizunoster/dfl-benchmark-training-fix-label-error)

# 如果用本文件生成的val目录, 作为训练过程中的验证集 去算eventAP,
# 会报错, 因为时间取ceil时 有些帧没保存?
# 可以用../working/pre_proc.py生成的val目录
    # (此时val/下的4个类别目录, 里面的图片就算从一个目录mv到另外一个, 也不影响eventAP吧?
    # 4个目录仅仅作为4分类的class label, 但我验证时, 直接用../input/dfl-bundesliga-data-shootout/train.csv)

# imgs4train/val, 不包含整段验证视频
    # 在label中, 上一个事件的end和下一个事件的start 之间的帧, 算eventAP时被忽略, 不论预测为哪个事件都不影响得分
    # 在比赛test set上推理时, 没有label, 所以每一帧都要推理,
    # 但在训练过程中验证时, 有label, 没必要管那些不算分的帧

# ----------------------------
#
# I found label errors in the baseline code.

# Fixing label errors give me a little boost on my score.
# Please see the description in loading label data for more information.
#
# !mkdir -p ../work
# !cd ../work && tar xfz ../input/dflfiles/timm.tgz

# ----------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.display import Video
import cv2
import math  # Added for fixed extract_images function

# ----------------------------
# # setting

debug = False

err_tol = {
    'challenge': [ 0.30, 0.40, 0.50, 0.60, 0.70 ],
    'play': [ 0.15, 0.20, 0.25, 0.30, 0.35 ],
    'throwin': [ 0.15, 0.20, 0.25, 0.30, 0.35 ]
}
video_id_split = {
    # 'val':[
    #     '3c993bd2_0',
    #     '3c993bd2_1',
    # ],
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

# ----------------------------
# # load label data

df = pd.read_csv("../input/dfl-bundesliga-data-shootout/train.csv")
additional_events = []
for arr in df.sort_values(['video_id','time','event','event_attributes']).values:
    if arr[2] in err_tol:
        tol = err_tol[arr[2]][0]/2
        additional_events.append([arr[0], arr[1]-tol, 'start_'+arr[2], arr[3]])
        additional_events.append([arr[0], arr[1]+tol, 'end_'+arr[2], arr[3]])
df = pd.concat([df, pd.DataFrame(additional_events, columns=df.columns)])
df = df[~df['event'].isin(event_names)]
df = df.sort_values(['video_id', 'time'])



# ## Major change
    # I also changed this part from the original code
        # because some output files are duplicated and labeled in different classes.
    # The original code performs a ¿time-based¿ increment in the innermost loop,
    # but my modification is to perform a ¿frame¿-based increment.
    # Examples of problems with the original code are as follows:

    #   ・
    # start                          (time=239.0, frame=5976)  ->labeled as backgraund
    #   ・                                                     ->bg

    # start + time_interval*10       (time=239.4, frame=5986)  ->frame 5986 is labeled as backgraund
    #   ・                                                     ->bg
    # start_event1                   (time=239.4, frame=5986)  ->frame 5986 is ¿also¿ labeled as event1

    #   ・                                                     ->event1
    # end_event1
    #   ・                                                     ->bg
    # start_event2                   (time=558.0, frame=13950) ->labeled as event2
    #   ・                                                     ->event2
    # start_event2 + time_interval*3 (time=558.1, frame=13953) ->frame 13953 is labeled as event2
    #   ・                                                     ->event2
    # end_event2                     (time=558.1, frame=13953) ->frame 13953 is ¿also¿ labeled as bg
    #   ・                                                     ->bg
    # end


def extract_training_images(args):
    video_id, split = args
    video_path = f"../input/dfl-bundesliga-data-shootout/train/{video_id}.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        TODO
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_interval = 1/fps

    df_video = df[df.video_id == video_id]
    if debug:
        df_video = df_video.head(10)
    print(split, video_id, df_video.shape)

    frmS_set = set()
    #crr_statu => bg, play, challenge, throwin
    arr = df_video[['time','event']].values
    for idx in range(len(arr)-1):
        # ¿Major changes from here¿
        crr_frame = int(math.ceil(arr[idx,0] * fps))
        nxt_frame = int(math.ceil(arr[idx+1,0] * fps))

        crr_event = arr[idx,1]
        #print(crr_time, nxt_time, crr_event)

        # crr_event = crr_event  为啥作者写这行? 忘记删掉?
        if crr_event == 'start':
            crr_status = 'bg'
        elif crr_event == 'end':
            # should use as bg?
            continue
        else:
            start_or_end, crr_status = crr_event.split('_', 1)
            if start_or_end == 'end':
                crr_status = 'bg'

        result_dir = f"../work/train_val__9frm_as_3frm/{split}/{crr_status}"
        # result_dir = f"../work/imgs4train_frmAStime/{split}/{crr_status}"
        # result_dir = f"../work/split_images/{split}/{crr_status}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)

        this_frame = crr_frame
        while this_frame < nxt_frame:
            frame_num = this_frame
            # "fix label error" 的作者加这2行, 验证自己没有label error?
            # (我把list改为dict, 快?)
            assert frame_num not in frmS_set
            frmS_set.add(frame_num)

            save_n_frmS_as_a_img = 1
            if save_n_frmS_as_a_img:
                #获取前中后各一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
                _, frm_left = cap.read()

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _, frame = cap.read()

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num+1)
                _, frame_right = cap.read()

                #
                frm_left_g    = cv2.cvtColor(frm_left    ,cv2.COLOR_BGR2GRAY)
                frame_g       = cv2.cvtColor(frame       ,cv2.COLOR_BGR2GRAY)
                frame_right_g = cv2.cvtColor(frame_right ,cv2.COLOR_BGR2GRAY)

                frame = cv2.merge([cv2.resize(frm_left_g    ,IMG_SIZE) ,
                                   cv2.resize(frame_g       ,IMG_SIZE) ,
                                   cv2.resize(frame_right_g ,IMG_SIZE) ,
                                  ])
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

            out_file = f'{result_dir}/{video_id}-{frame_num:06}.jpg'
            cv2.imwrite(out_file, frame)


            # print(out_file, arr[idx], arr[idx+1], this_frame)

            if crr_status == 'bg':
                this_frame += 10
            else:
                this_frame += 1

# !rm -rf ../work/split_images/
for split in video_id_split:
    video_ids = video_id_split[split]
    for video_id in video_ids:
        extract_training_images([video_id, split])

