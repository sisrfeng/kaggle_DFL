exp_PATH=use_bce_loss__val_on_eventAP__Nov_22
t exp/${exp_PATH}
                    # ../work/imgs4train     \
p train_timm.py                                \
    --amp                                      \
    --pretrained                               \
    --num_classes   4                          \
    --model         tf_efficientnet_b5_ap      \
    --epochs        100                        \
    --log_interval  5000                       \
    --batch_size    4                          \
    --which_metric  event_AP                   \
    --bce_loss                                 \
    --data_dir      ../work/imgs4train__debug  \
    --output        ./exp                      \
    --experiment    $exp_PATH


# cd ../input/upload-wf     ;  \
# python train_timm.py  ../../work/imgs4train     \
    # --experiment    ../../working/use_bce_loss
    #
# --which-metric  loss                   \
    # Acc@N doesn't mean much, as you mentioned.
    # It would be better to use loss(you can use --eval-metric loss option for this) or the original metrics function.


