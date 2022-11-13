cd ../input/upload-wf     ;  \
b train_timm.py  ../../work/imgs4train     \
    --amp                                  \
    --pretrained                           \
    --num-classes   4                      \
    --model         tf_efficientnet_b5_ap  \
    --epochs        100                    \
    --out           ./                     \
    --log-interval  5000                   \
    --batch-size    8                      \
    --which-metric  loss                   \
    --experiment    ../../working/exp_metric_loss
    # --initial-checkpoint ./wf_try/checkpoint-19.pth.tar
    #
# --which-metric  loss                   \
    # Acc@N doesn't mean much, as you mentioned.
    # It would be better to use loss(you can use --eval-metric loss option for this) or the original metrics function.

