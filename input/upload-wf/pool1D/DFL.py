if 1:
    from torch.utils.data import Dataset
    import numpy as np
    import random
    import os
    import time
    from tqdm import tqdm
    import torch
    import logging
    import json
    import zipfile
    import pandas as pd

if 'const':
    class2id = {"play":0, "throwin":1, "challenge":2}
    id2class = {0:"play", 1:"throwin", 2:"challenge"}
    EVENT_DICTIONARY_V1 = {"card": 0, "subs": 1, "soccer": 2}


    FPS              = 25
    event_names      = ['challenge', 'throwin', 'play', 'background']

def getListGames(split):
    listgames = list()
    if not isinstance(split, list):
        split = [split]
    for spl in split:
        if spl == 'train':
            for file in os.listdir(os.path.join(P_feat, 'train')):
                name, ext = os.path.splitext(file)
                if ext=='.npy':
                    listgames.append(os.path.join('train', name))
        if spl == 'test' or spl =='valid':
            for file in os.listdir(os.path.join(P_feat, 'valid')):
                name, ext = os.path.splitext(file)
                if ext=='.npy':
                    listgames.append(os.path.join('valid', name))
        if spl == 'challenge':
            for file in os.listdir(os.path.join(P_feat, 'challenge')):
                name, ext = os.path.splitext(file)
                if ext=='.npy':
                    listgames.append(os.path.join('challenge', name))
    return listgames


def feats_game2clip(feat1Game,
                    clip_length                ,
                    stride                     ,
                    off     = 0                ,
                    padding = "replicate_last" ,
                   ):

    if padding =="zeropad":
        pad       = feat1Game.shape[0] - int(feat1Game.shape[0]/stride)*stride
        feat1Game = torch.nn.ZeroPad2d( ( 0, 0, clip_length - pad, 0 ) )(feat1Game)

    idx = torch.arange( start=0, end=feat1Game.shape[0]-1, step=stride )
    idxs = []
    for i in torch.arange(-off, clip_length-off):
        idxs.append(idx+i)
    idx = torch.stack(idxs, dim=1)

    if padding=="replicate_last":
        idx = idx.clamp(0, feat1Game.shape[0]-1)
    # print(idx)
    return feat1Game[idx,...]

import glob

class Clip_of_gameS(Dataset):
    def __init__(self,
                 P_feat                        ,
                 SPLIT                = 'Test' ,
                 input_FpS            = 25     ,
                 pool_win_frms_radius = 5     ,
                ):
        self.P_feat   = P_feat

        self.input_FpS = input_FpS
        self.pool_win_frms = 1 + 2 * pool_win_frms_radius

        self.SPLIT=SPLIT
        self.dict_event = class2id
        self.num_classes = 3

        if SPLIT == 'Test':
            gameS_long = sorted(glob.glob('../../dfl-bundesliga-data-shootout/test/*'))
            self.gameS = []
        #
            for name in gameS_long:
                want = name.split('/')[-1].split('.')[0]
                self.gameS.append(want)
        else:
            self.gameS   = [ '3c993bd2_0', '3c993bd2_1' ]


    def __getitem__(self, index):
        # import pudb
        # pu.db
        feat_npy    = np.load(f'../../../working/{self.SPLIT}_feat2048.npy')
        fNames_infer = np.load(f'../../../working/{self.SPLIT}_nameS__vdo_img.npy')

        #排序
        sorted_indices = np.argsort(fNames_infer)

        feat_npy       = feat_npy[   sorted_indices]
        fNames_infer    = fNames_infer[sorted_indices].tolist()

        feat_npy = feat_npy.reshape( -1, feat_npy.shape[-1] )

        game = self.gameS[index]
        frmS_1game = []
        for i, fName in enumerate(fNames_infer):
            if game in fName:
                frmS_1game.append(i)
        # feat1Game = feat_npy[frmS_1game][0]
        feat1Game = feat_npy[frmS_1game]

        feat1Game = feat1Game.reshape(-1, feat1Game.shape[-1])

        # Load labels
        label1Clip = np.zeros((feat1Game.shape[0], self.num_classes))

        # check if annoation exists
        if os.path.exists(os.path.join(self.P_feat, self.gameS[index]+'.json')):
            labels = json.load(open(os.path.join(self.P_feat, self.gameS[index]+'.json')))

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                frame = round(self.input_FpS*time)

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]
                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                frame = min(frame, feat1Game.shape[0]-1)
                label1Clip[frame][label] = value

        feat1Clip = feats_game2clip(torch.from_numpy(feat1Game),
                                    clip_length = self.pool_win_frms        ,
                                    stride      = 1                         ,
                                    off         = int( self.pool_win_frms / 2 ) ,
                                   )

        return self.gameS[index], feat1Clip, label1Clip



    def __len__(self):
        return len(self.gameS)
