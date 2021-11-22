# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='ResNet18', norm_type='sync_batchnorm'),
    decode_head=dict(
        type='PSPHead',
        in_channels=512,
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(rampup_length=4000),
    test_cfg=dict(mode='whole'),
)
