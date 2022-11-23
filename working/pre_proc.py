# 来自0.249分的baseline
if 1:
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sn
    from IPython.display import Video
    import cv2


events2tol = {
    'challenge' : [ 0.30, 0.40, 0.50, 0.60, 0.70 ],
    'play'      : [ 0.15, 0.20, 0.25, 0.30, 0.35 ],
    'throwin'   : [ 0.15, 0.20, 0.25, 0.30, 0.35 ]
}

splits2vdoIds = {
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

df = pd.read_csv("../input/dfl-bundesliga-data-shootout/train.csv")
event3x2 = []
for row in df.sort_values( ['video_id', 'time', 'event', 'event_attributes']).values:
    if row[2] in events2tol:
        tol =   events2tol[ row[2] ][0]   / 2
                                  # 改用其他tolerance?
        event3x2.append( [  row[0], row[1]-tol, f'start_{row[2]}', row[3]  ] )
        event3x2.append( [  row[0], row[1]+tol,   f'end_{row[2]}', row[3]  ] )
df = pd.concat( [df,
                 pd.DataFrame(event3x2, columns=df.columns),
                ] )
df_2P6 = df[  ~df['event'].isin( ['challenge', 'throwin', 'play']) ]  # 2 plus 6 classes: start + end   +  2x3 classes
            # ~ : 取反
df_2P6 = df_2P6.sort_values(['video_id', 'time'])

def vdo2img(vdo_id, split):
    video_path      = f"../input/dfl-bundesliga-data-shootout/train/{vdo_id}.mp4"
                                                           # 把官方的train划分出train和val
    # video_path      = f"../input/dfl-bundesliga-data-shootout/{split}/{vdo_id}.mp4"
    cap             = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('错错_______________________________')

    fps           = cap.get(cv2.CAP_PROP_FPS)
    frm2frm = 1 / fps

    df_2P6_a_vdo = df_2P6[ df_2P6.video_id == vdo_id ]
    #  Debug:
    # df_2P6_a_vdo = df_2P6_a_vdo.head(10)

    row = df_2P6_a_vdo[ ['time','event'] ].values
    for idx in range( len(row) - 1 ):
        this_event = row[idx   , 1]

        this_time  = row[idx   , 0]
        next_time  = row[idx+1 , 0]
        # Label is asigned as follows:
            #       ・
            #     start
            #       ・                   -> bg
            #     event1 - tolerances
            #       ・                   -> event1
            #     event1 + tolerances
            #       ・                   -> bg


            #     event2 - tolerances
            #       ・                   -> event2
            #     event2 + tolerances
            #       ・                   -> bg
            #     end
                 # ・
                 # ・
                 # ・
                 # ・
            #       ・                   -> not used  (should use as bg? 但用上的话 bg就太多了? 类别不平衡?)
            #     start

        if this_event == 'start':
            event_1in4 = 'bg'
        elif this_event == 'end':
            continue
            # not used  (should use as bg? 但用上的话 bg就太多了? 类别不平衡?)
            # 下面frm2frm * 10 就是为了少要些bg?
        else:
            start_or_end, event_1in4 = this_event.split('_', 1)
            if start_or_end == 'end':
                event_1in4 = 'bg'

        P_img = f"../work/imgs4train/{split}/{event_1in4}"
        if not os.path.exists(P_img):
            os.makedirs( P_img, exist_ok=True )

        while this_time < next_time:
            frm_id = int( this_time * fps )

            cap.set( cv2.CAP_PROP_POS_FRAMES, frm_id )
            _ , frm   = cap.read()
            # out_file = f'{P_img}/{vdo_id}_{frm_id:06}.jpg'  日红把人家的¿-¿改成了¿_¿ ,导致make_sub时出错
            out_file = f'{P_img}/{vdo_id}-{frm_id:06}.jpg'
            cv2.imwrite(out_file, frm)  # 如果图片存在 会重新write?
            # print('out_file, row[idx], row[idx+1], this_time:')
            # print(out_file, row[idx], row[idx+1], this_time)

            if event_1in4 == 'bg':
                this_time += frm2frm * 10
                                      #   跳过10帧,  10是调参得到的?
            else:
                this_time += frm2frm


for split in splits2vdoIds:
    vdo_idS = splits2vdoIds[split]
    for vdo_id in vdo_idS:
        vdo2img(vdo_id, split)

print('pre processing:  done')


# 后续的放到train.zsh里了:
