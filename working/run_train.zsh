exp_PATH=batchSize8__Correct_label___ori_input_size__Mixup__$(date +"%-d号%H点")
echo ${exp_PATH}
# t可以改成rm , 我这里的t是扔进垃圾桶的意思, 避免旧实验干扰
t exp/${exp_PATH}

# p是改成python就行 (我把p设成了python和一些智能化的操作alias)
# p train_timm.py   ../work/imgs4train__debug  \
# p train_timm.py   ../work/imgs4train         \
p train_timm.py   ../work/correct_images4train         \
    --amp                                  \
    --pretrained                           \
    --num_classes   4                      \
    --model         tf_efficientnet_b5_ap  \
    --epochs        100                    \
    --log_interval  5000                   \
    --batch_size    8                      \
    --which_metric  event_AP               \
    --bce_loss                             \
    --mixup         0.5                    \
    --output        ./exp                  \
    --experiment    $exp_PATH

# tito:
    # 1. titanRTX x2 environment it only took 6 hours for 40 epochs

     # 2. CrossEntropy is used here,
     # 可以试: Binary CrossEntropy and making only 3 Targets with background being [0,0,0]
     # Or, MSE(Regression) may wark too.
        # In that case, a Gaussian distribution might work for GT.
        # exp(-(time - event_time)^2/2tolerances^2).
  #
  #
# 关于--which-metric
    # Acc@N doesn't mean much, as you mentioned.
    # It would be better to use loss(you can use --eval-metric loss option for this)
    # or the original metrics function.


# 模型融合
# !cd ../work/timm/pytorch-image-models && \
    # python ./avg_checkpoints.py --input output/train/dfl-benchmark-training-fix-extract-images \
    # --output /kaggle/working/tf_efficientnet_b5_ap-456-fix.pt




