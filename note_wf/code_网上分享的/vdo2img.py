# import debugpy
# debugpy.listen(("localhost", 5678))  #  start the debug adapter,
import cv2
import imageio
import pandas as pd
from tqdm import tqdm

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')


def save_video(start_time, end_time, camera, save_path, FPS, size):
    start_index = int(start_time * FPS)
    end_index = int(end_time * FPS)
    video = cv2.VideoWriter(save_path, fourcc, FPS, size)
    for i in range(start_index, end_index):
        frame = camera.get_data(i)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()


def save_frame(event_time, camera, save_path, FPS):
    event_index = int(event_time * FPS)
    frame = camera.get_data(event_index)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path,frame)


df = pd.read_csv('../input/dfl-bundesliga-data-shootout/train.csv')
current_video = ''
camera = None
# breakpoint()  # or debugpy.breakpoint()
# 加了DAP的东西, 导致pudb的断点不会停

for idx, row in tqdm(df.iterrows()):
    video_id, ts, event, event_attributes = row['video_id'], row['time'], row['event'], row['event_attributes']
    if event == 'start':
        if not video_id == current_video:
            if camera is not None:
                camera.close()
            current_video = video_id
            video_path = r'../input/dfl-bundesliga-data-shootout/train/%s.mp4' % (current_video)
            camera = imageio.get_reader(video_path, "ffmpeg")
            metadata = camera.get_meta_data()
            print(metadata)
            FPS = metadata['fps']
            size = metadata['size']
        start_time = ts
        event_list = []
        event_attributes_list = []
    elif event == 'end':
        # end_time = ts
        # if len(event_list) == 1:
        #     save_path = '../output/%s/%s_%.3f_%s.mp4' % (event_list[0], video_id, start_time, '-'.join(event_attributes_list[0]))
        # elif len(set(event_list)) == 1:
        #     save_path = '../output/%s/%s_%.3f_%s.mp4' % (
        #         event_list[0], video_id, start_time, '_'.join(['-'.join(event_attributes) for event_attributes in event_attributes_list]))
        # else:
        #     name_list = []
        #     for event, event_attributes in zip(event_list, event_attributes_list):
        #         name_list.append('%s-%s' % (event, '-'.join(event_attributes)))
        #     save_path = '../output/%s_%.3f_%s.mp4' % (video_id, start_time, '_'.join(name_list))
        # # print(save_path)
        # save_video(start_time, end_time, camera, save_path, FPS, size)
        pass
    else:
        event_attributes = event_attributes.replace('[', '').replace(']', '').replace('\'', '').split(',')
        event_list.append(event)
        event_attributes_list.append(list(event_attributes))
        save_path = '../output/img/%s/%s_%.3f_%s_%.3f.jpg' % (event, video_id, start_time, '-'.join(event_attributes), ts)
        save_frame(ts,camera,save_path,FPS)
