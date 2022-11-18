# * [Overview](#1)
# * [EVENT: throwin](#2)
# * [EVENT: play](#3)
# * [EVENT: challenge](#4)
# * [Process Video](#5)

# !pip install moviepy -q

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from IPython.display import Video

df = pd.read_csv("../input/dfl-bundesliga-data-shootout/train.csv")

print('df是')
print(df)


plt.figure(figsize=(16, 5))
sn.countplot(data=df, y="event")


plt.figure(figsize=(16, 5))
sn.countplot(data=df, y="video_id")


plt.figure(figsize=(16, 5))
sn.countplot(data=  df[  (df["event"] != "start") & (df["event"] != "end")  ],    y="video_id")

# %% [markdown] {"execution":{"iopub.status.busy":"2022-07-29T22:57:53.824078Z","iopub.execute_input":"2022-07-29T22:57:53.824481Z","iopub.status.idle":"2022-07-29T22:57:53.832435Z","shell.execute_reply.started":"2022-07-29T22:57:53.824449Z","shell.execute_reply":"2022-07-29T22:57:53.831317Z"}}
# <a id="2"></a>
# <h2 style='background:green; border:0; color:white'><center>EVENT: throwin<center><h2>


df_throwin = df[ df["event"] == "throwin"].reset_index()


plt.figure(figsize=(16, 3))
sn.countplot(data=df_throwin, y="event_attributes")


plt.figure(figsize=(16, 6))
sn.countplot(data=df_throwin, x="video_id", hue="event_attributes")


def vis_event(row, before=5, after=5):
    print(row["event_attributes"])
    filename = f"test_{row['index']}.mp4"
    ffmpeg_extract_subclip(
        f"../input/dfl-bundesliga-data-shootout/train/{row['video_id']}.mp4",
        int(row['time']) - before,
        int(row['time']) + after,
        targetname=filename,
    )

    return Video(filename, width=800)

# %% [markdown]
# #### ["pass"]
#
# Pass, defined to be any attempt to switch ball control to another team member that doesn’t satisfy cross definition.


vis_event(df_throwin.iloc[0])

# %% [markdown]
# #### ["cross"]
#
# Whether a play is a Cross depends upon the positions of the acting player and of the possible recipient. The player playing the cross must be located approx. inside one of the four crossing zones. The four zones are marked by the touchlines, the extended sides of the penalty area, the goal lines and the imaginary quarter-way lines, which would be drawn at a quarter of the length of the pitch parallel to the half-way line (see figure below). The possible cross recipient must be located approx. inside the penalty area. Furthermore, the distance of the ball played must be medium-length (from 10 to 30 meters) or long (more than 30 meters) and the height of the ball played must be high (played above knee height). In order to classify a ball played as a cross if the ball is blocked by an opposing player, it is not the actual height or distance travelled that is decisive, but the intended height or distance.
#
# ![](https://i.imgur.com/dvHxE1s.png)


vis_event(df_throwin.iloc[2])

# %% [markdown]
# <a id="3"></a>
# <h2 style='background:green; border:0; color:white'><center>EVENT: play<center><h2>


df_play = df[df["event"] == "play"].reset_index()


plt.figure(figsize=(16, 3))
sn.countplot(data=df_play, y="event_attributes")


plt.figure(figsize=(16, 6))
sn.countplot(data=df_play, x="video_id", hue="event_attributes")
plt.ylim(0, 50);


df_play

# %% [markdown]
# ### ["pass", "openplay"]
#


vis_event(df_play.iloc[0], before=2, after=2)

# %% [markdown]
# ### ["cross", "openplay"]


vis_event(df_play.iloc[15], before=2, after=2)

# %% [markdown]
# ### ["pass", "freekick"]
#
# A Free Kick refers to a situation where the Play is executed to restart the game after the referee had stopped it due to an infringement of the rules. The ball must be kicked and be stationary on the ground when it's kicked off.


vis_event(df_play.iloc[27], before=2, after=2)

# %% [markdown]
# ### ["cross", "freekick"]


vis_event(df_play.iloc[1105], before=2, after=2)

# %% [markdown]
# ### ["pass", "corner"]
#
# A Corner Kick refers to a situation where the Play is executed to restart the game after the ball went out of play over the goal line following the touch of the defending team player. The ball must be kicked from the closest field corner and be stationary on the ground when it’s kicked off.


vis_event(df_play.iloc[1955], before=2, after=2)

# %% [markdown]
# ### ["cross", "corner"]


vis_event(df_play.iloc[108], before=2, after=2)

# %% [markdown] {"execution":{"iopub.status.busy":"2022-07-29T23:28:45.137203Z","iopub.execute_input":"2022-07-29T23:28:45.137594Z","iopub.status.idle":"2022-07-29T23:28:45.14609Z","shell.execute_reply.started":"2022-07-29T23:28:45.137559Z","shell.execute_reply":"2022-07-29T23:28:45.144241Z"}}
# <a id="4"></a>
# <h2 style='background:green; border:0; color:white'><center>EVENT: challenge<center><h2>


df_challenge = df[df["event"] == "challenge"].reset_index()


plt.figure(figsize=(16, 3))
sn.countplot(data=df_challenge, y="event_attributes")


plt.figure(figsize=(16, 5))
sn.countplot(data=df_challenge, x="video_id", hue="event_attributes")


df_challenge["event_attributes"].value_counts()


for i, x in df_challenge.iterrows():
    if x["event_attributes"] == "['possession_retained']":
        print(i, x)
        break

# %% [markdown]
# ### ["opponent_rounded"]
#
# Opponent rounded: a player in ball control stays in ball control after the challenge, having left the opposing player behind him. Situations where the oppone nt is not able to gain possession (e.g. when the ball is “flicked” over the opponent) are also to be recorded as challenges.


vis_event(df_challenge.iloc[11], before=3, after=3)

# %% [markdown]
# ### ["ball_action_forced"]
#
# Ball action carried out: applies when none of the players involved in the challenge are in ball control at the start of the challenge (e.g. aerial challenges, challenges for the first touch of the ball) and one player determines the direction of the ball at the end of the challenge.


vis_event(df_challenge.iloc[0], before=3, after=3)

# %% [markdown]
# ### ["fouled"]
#
# Fouled: the referee called a foul.


vis_event(df_challenge.iloc[8], before=3, after=3)

# %% [markdown]
# ### ["opponent_dispossessed"]
#
# Opponent dispossessed: a player not in ball control dispossesses the opposing player in ball control.


vis_event(df_challenge.iloc[1], before=3, after=3)

# %% [markdown]
# ### ["challenge_during_ball_transfer"]
#
# Challenge during release of the ball: applies when shots or balls played are forced or blocked during challenges. A challenge is only recorded, if the ball played or shot travels through the area that the defending player is attempting to cover from a tactical perspective. All other cases are not recorded as challenges.


vis_event(df_challenge.iloc[45], before=3, after=3)

# %% [markdown] {"execution":{"iopub.status.busy":"2022-07-30T22:06:34.545825Z","iopub.execute_input":"2022-07-30T22:06:34.546254Z","iopub.status.idle":"2022-07-30T22:06:34.554853Z","shell.execute_reply.started":"2022-07-30T22:06:34.546195Z","shell.execute_reply":"2022-07-30T22:06:34.553079Z"}}
# ### ["possession_retained"]
#
# Possession retained during challenge: applies when one of the players involved in the challenge has certain ball control at the start of the challenge and manages to retain it, despite the efforts to dispossess him of the opponent involved in the challenge.


vis_event(df_challenge.iloc[2], before=3, after=3)

# %% [markdown]
# <a id="5"></a>
# <h2 style='background:green; border:0; color:white'><center>Process Video<center><h2>


import cv2


vidcap = cv2.VideoCapture("test_130.mp4")


success, image = vidcap.read()


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(16, 8))
plt.imshow(image);


vidcap = cv2.VideoCapture("test_130.mp4")

fps = vidcap.get(cv2.CAP_PROP_FPS)
print(f"fps: {fps}")
frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"frame_count: {frame_count}")

count = 0
success, image = vidcap.read()
while success:
    count += 1
    success, image = vidcap.read()

print(f"{count} frames.")


plt.figure(figsize=(16, 48))

vidcap = cv2.VideoCapture("test_130.mp4")
np_video = []

success, image = vidcap.read()
count = 0
step = 10
while success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    np_video.append(image)
    if not count % step:
        plt.subplot(8, 2, count // step + 1)
        plt.imshow(image)
        plt.axis("off")
    success, image = vidcap.read()

    count += 1


from matplotlib import animation, rc
rc('animation', html='jshtml')


def create_animation(ims):
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000 // 24)


create_animation(np_video)


test_image = np_video[0]

plt.figure(figsize=(16, 8))
plt.imshow(test_image)

plt.figure(figsize=(16, 8))
gray_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image, cmap="gray")

plt.figure(figsize=(16, 8))
plt.imshow(gray_image >= 182, cmap="gray")


# https://towardsdatascience.com/football-players-tracking-identifying-players-team-based-on-their-jersey-colors-using-opencv-7eed1b8a1095

color_list = ["red", "blue", "white"]

boundaries = [
    ((17, 15, 75), (50, 56, 200)),
    ((43, 31, 4), (250, 88, 50)),
    ((187, 169, 112), (255, 255, 255)),
]


for color, boundary in zip(color_list, boundaries):
    mask = cv2.inRange(test_image, boundary[0], boundary[1])

    output = cv2.bitwise_and(test_image, test_image, mask=mask)

    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(test_image)
    plt.subplot(2, 2, 3)
    plt.imshow(output)
    plt.subplot(2, 2, 2)
    plt.imshow(mask)
    plt.show()


