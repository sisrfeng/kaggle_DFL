if 'import':
    import logging
    import os
    import zipfile
    import sys
    import json
    import time
    from tqdm import tqdm
    import torch
    import numpy as np

    import sklearn
    import sklearn.metrics
    from sklearn.metrics import average_precision_score

id2class = {0:"play", 1:"throwin", 2:"challenge"}


def test(loader, model, model_name):

    model.eval()

    all_labels = []
    all_outputs = []
    for i, (feats, labels) in  enumerate(loader):
        feats = feats.cuda()
        # labels = labels.cuda()
        # feats=feats.unsqueeze(0)

        # compute output
        output = model(feats)

        all_labels.append(labels.detach().numpy())
        all_outputs.append(output.cpu().detach().numpy())


    AP = []
    for i in range(1, loader.dataset.num_classes+1):
        AP.append(average_precision_score(np.concatenate(all_labels)
                                          [:, i], np.concatenate(all_outputs)[:, i]))

    # t.set_description()
    # print(AP)
    mAP = np.mean(AP)
    print(mAP, AP)

    return mAP

def testSpotting(loader,
                 model,
                 model_name,
                 overwrite      = True,
                 NMS_secs3event = [999,999,999],
                               #三类的时间窗口，play，throwin, challenge
                 NMS_threshold  = 0.5,
                 SPLIT          = 'Test',
                ):
    model.eval()
    # probS__all_vdo = []
    # import pudb; pu.db
    os.makedirs( f'../../../work/probS_npy/{SPLIT}', exist_ok=True)
    for i, (game_ID, feat1Clip, label1Clip) in  enumerate(loader):
        # Batch size of 1
        # import pudb; pu.db
        game_ID     = game_ID[0]
        feat1Clip   = feat1Clip.squeeze(0)
        label1Clip  = label1Clip.float().squeeze(0)


        # Compute the output for batches of frames
        BS = 256
        probS1vdo = []
        for b in range( int(np.ceil( len(feat1Clip) / BS )) ):
            start_frame = BS * b
            end_frame   = BS * (b + 1) \
                            if BS *  (b+1) < len(feat1Clip) \
                            else \
                         len(feat1Clip)
            feat = feat1Clip[start_frame:end_frame].cuda()
            probS1vdo.append( model(feat).cpu().detach().numpy() )

        probS1vdo = np.concatenate(probS1vdo,axis=0)
        # probS__all_vdo = np.concatenate(probS__all_vdo, axis=0)
        np.save(f'../../../work/probS_npy/{SPLIT}/{game_ID}.npy', probS1vdo)
        # probS__all_vdo.extend(probS1vdo)

    # probS__all_vdo = np.concatenate(probS__all_vdo, axis=0)

    # np.save('probS__all_vdo.npy', probS__all_vdo )
    # return  np.concatenate(probS__all_vdo)
    return 1
#
