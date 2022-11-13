cd ../train_git
p train.py ../../work/imgs4train           \
    --amp                                  \
    --pretrained                           \
    --num-classes   4                      \
    --model         tf_efficientnet_b5_ap  \
    --epochs        100                    \
    --out           ./                     \
    --log-interval  5000                   \
    --batch-size    8                      \
    --which-metric  loss                   \
    --experiment    corrected_label


# 模型融合
# !cd ../work/timm/pytorch-image-models && \
    # python ./avg_checkpoints.py --input output/train/dfl-benchmark-training-fix-extract-images \
    # --output /kaggle/working/tf_efficientnet_b5_ap-456-fix.pt



