exp_PATH=new_input_size%%mixup%%$(date +"%-d号%H点")
# asdf
echo ${exp_PATH}
# t可以改成rm , 我这里的t是扔进垃圾桶的意思, 避免旧实验干扰
t exp/${exp_PATH}

# p train_timm.py   ../work/imgs4train__debug  \
#
# p是改成python就行 (我把p设成了python和一些智能化的操作alias)
p train_timm.py   ../work/imgs4train         \
    --amp                                  \
    --pretrained                           \
    --num_classes   4                      \
    --model         tf_efficientnet_b5_ap  \
    --epochs        100                    \
    --log_interval  5000                   \
    --batch_size    2                      \
    --which_metric  event_AP               \
    --bce_loss                             \
    --input_size 3 540 960                 \
    --mixup         0.5                    \
    --output        ./exp                  \
    --experiment    $exp_PATH


# cd ../input/upload-wf     ;  \
# python train_timm.py  ../../work/imgs4train     \
    # --experiment    ../../working/use_bce_loss
    #
# --which-metric  loss                   \
    # Acc@N doesn't mean much, as you mentioned.
    # It would be better to use loss(you can use --eval-metric loss option for this) or the original metrics function.


