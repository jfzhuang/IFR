# dataset settings
data_root = 'data/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (200, 240)
train_pipeline = [
    dict(type='LoadImageFromFile_Semi'),
    dict(type='LoadAnnotations_Semi'),
    dict(type='RandomCrop_Semi', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip_Semi', prob=0.5),
    dict(type='PhotoMetricDistortion_Semi'),
    dict(type='Normalize_Semi', **img_norm_cfg),
    dict(type='DefaultFormatBundle_Semi'),
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
        img_scale=(480, 360),
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
        type='CamVidSemiDataset',
        data_root=data_root,
        img_dir='CamVid/video_image_down_2x',
        ann_dir='CamVid/label_down_2x',
        split='CamVid/splits/trainval_video_1-30.txt',
        split_unlabeled='CamVid/splits/trainval_video_1-1.txt',
        pipeline=train_pipeline,
    ),
    val=dict(
        type='CamVidDataset',
        data_root=data_root,
        img_dir='CamVid/video_image_down_2x',
        ann_dir='CamVid/label_down_2x',
        split='CamVid/splits/test_image.txt',
        pipeline=test_pipeline,
    ),
)
