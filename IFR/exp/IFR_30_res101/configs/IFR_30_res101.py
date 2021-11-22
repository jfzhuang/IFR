_base_ = [
    '../../../configs/_base_/models/pspnet_r101-d8.py',
    '../../../configs/_base_/datasets/cityscapes_semi.py',
    '../../../configs/_base_/default_runtime.py',
    '../../../configs/_base_/schedules/schedule_semi.py',
]

model = dict(
    type='IFR',
    temperature=0.5,
    weight_re_labeled=1e-2,
    weight_re_unlabeled=1e-3,
    weight_re_strong=0.1,
)
seed = 666
deterministic = True
