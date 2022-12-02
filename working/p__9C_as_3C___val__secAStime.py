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
    # 'train':[
    #      '1606b0e6_0',
    #      '1606b0e6_1',
    #      '35bd9041_0',
    #      '35bd9041_1',
    #      '407c5a9e_1',
    #      '4ffd5986_0',
    #      '9a97dae4_1',
    #      'cfbe2e94_0',
    #      'cfbe2e94_1',
    #      'ecf251d4_0',
    # ]
}

df = pd.read_csv( "../input/dfl-bundesliga-data-shootout/train.csv" )
event3x2 = []
for row in df.sort_values( ['video_id', 'time', 'event', 'event_attributes']).values:
    if row[2] in events2tol:
        tol =   events2tol[row[2]][0]   / 2
                                  # 改用其他tolerance?
        event3x2.append( [  row[0], row[1]-tol, f'start_{row[2]}', row[3]  ] )
        event3x2.append( [  row[0], row[1]+tol,   f'end_{row[2]}', row[3]  ] )
df = pd.concat( [df,
                 pd.DataFrame(event3x2, columns=df.columns),
                ] )
df_2P6 = df[  ~df['event'].isin( ['challenge', 'throwin', 'play']) ]  # 2 plus 6 classes: start + end   +  2x3 classes
              # ~ : 取反, 仍掉play等, 只要start_play和end_play等
df_2P6 = df_2P6.sort_values(['video_id', 'time'])

def vdo2img(vdo_id, split):
    cap = cv2.VideoCapture( f"../input/dfl-bundesliga-data-shootout/train/{vdo_id}.mp4")
    if not cap.isOpened():
        print('错了...')

    fps     = cap.get(cv2.CAP_PROP_FPS)
    frm2frm = 1 / fps

    df_2P6_a_vdo = df_2P6[ df_2P6.video_id == vdo_id ]
    #  Debug:
    # df_2P6_a_vdo = df_2P6_a_vdo.head(10)

    frmS_set = set()
    row = df_2P6_a_vdo[ ['time','event'] ].values
    for idx in range( len(row) - 1 ):
        this_event = row[idx   , 1]

        this_time  = row[idx   , 0]
        next_time  = row[idx+1 , 0]
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

        P_img = f"../work/train_val__9C_as_3C/{split}/{event_1in4}"
        # P_img = f"../work/imgs4train_frmAStime/{split}/{event_1in4}"
        # P_img = f"../work/imgs4train/{split}/{event_1in4}"

        if not os.path.exists(P_img):
            os.makedirs( P_img, exist_ok=True )

        while this_time < next_time:
            frm_id = int( this_time * fps )
            # "fix label error" 的作者加这2行, 验证自己没有label error?
            # (我把list改为dict, 快?)
            if  frm_id not in frmS_set:
                frmS_set.add(frm_id)
            else:
                print('重复了:', frm_id)

            save_n_frmS_as_a_img = 1
            if save_n_frmS_as_a_img:
                #获取前中后各一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id-1)
                _, frm_left = cap.read()

                cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
                _, frm = cap.read()

                cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id+1)
                _, frm_right = cap.read()

                #
                frm_left_g    = cv2.cvtColor(frm_left    ,cv2.COLOR_BGR2GRAY)
                frm_g       = cv2.cvtColor(frm       ,cv2.COLOR_BGR2GRAY)
                frm_right_g = cv2.cvtColor(frm_right ,cv2.COLOR_BGR2GRAY)

                frm = cv2.merge([ frm_left_g, frm_g, frm_right_g ])
            else:
                cap.set( cv2.CAP_PROP_POS_FRAMES, frm_id )
                ret, frm = cap.read()

            # out_file = f'{P_img}/{vdo_id}_{frm_id:06}.jpg'  日红把人家的¿-¿改成了¿_¿ ,导致make_sub时出错
            out_file = f'{P_img}/{vdo_id}-{frm_id:06}.jpg'
            cv2.imwrite(out_file, frm)  # 如果图片存在 会重新write?

            if event_1in4 == 'bg':
                this_time += frm2frm * 10  #  跳过10帧,  10是调参得到的?
            else:
                this_time += frm2frm

for split in splits2vdoIds:
    vdo_idS = splits2vdoIds[split]
    for vdo_id in vdo_idS:
        vdo2img(vdo_id, split)    #
