# dataset settings
data_root = 'data/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 512)
train_pipeline = [
    dict(type='LoadImageFromFile_Both_v2'),
    dict(type='LoadAnnotations_Both'),
    dict(type='RandomCrop_Both_v2', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_Both_v2', prob=0.5),
    dict(type='PhotoMetricDistortion_Both_v2'),
    dict(type='Normalize_Both_v3', **img_norm_cfg),
    dict(type='DefaultFormatBundle_Both_v3'),
    dict(
        type='Collect',
        keys=['img_v0_0', 'img_v0_1', 'img_v0_1_s', 'img_v1_0', 'img_v1_1', 'img_v1_1_s', 'gt'],
        meta_keys=(),
    ),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CityscapesSemiDataset',
        data_root=data_root,
        img_dir='cityscapes/leftImg8bit_sequence_down_2x/train',
        ann_dir='cityscapes/gtFine_down_2x/train',
        split='splits/train_unsup_1-30.txt',
        split_unlabeled='splits/train_unsup_all.txt',
        pipeline=train_pipeline,
    ),
    val=dict(
        type='CityscapesDataset',
        data_root=data_root,
        img_dir='cityscapes/leftImg8bit_sequence_down_2x/val',
        ann_dir='cityscapes/gtFine_down_2x/val',
        split='splits/val.txt',
        pipeline=test_pipeline,
    ),
)
