import numpy as np

label_colours_viper = [
    (70, 130, 180),
    (128, 64, 128),
    (244, 35, 232),
    (152, 251, 152),
    (87, 182, 35),
    (35, 142, 35),
    (70, 70, 70),
    (153, 153, 153),
    (190, 153, 153),
    (150, 20, 20),
    (250, 170, 30),
    (220, 220, 0),
    (180, 180, 100),
    (173, 153, 153),
    (168, 153, 153),
    (81, 0, 21),
    (81, 0, 81),
    (220, 20, 60),
    (0, 0, 230),
    (0, 0, 142),
    (0, 80, 100),
    (0, 60, 100),
    (0, 0, 70),
    (0, 0, 0),
]

label_colours = [
    (128, 64, 128),
    (244, 35, 231),
    (69, 69, 69),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 29),
    (219, 219, 0),
    (106, 142, 35),
    (152, 250, 152),
    (69, 129, 180),
    (219, 19, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 69),
    (0, 60, 100),
    (0, 79, 100),
    (0, 0, 230),
    (119, 10, 32),
    (0, 0, 0),
]


def decode_labels_viper(mask):
    h, w = mask.shape
    mask[mask == 255] = 23
    color_table = np.array(label_colours_viper, dtype=np.float32)
    out = np.take(color_table, mask, axis=0)
    out = out.astype(np.uint8)
    out = out[:, :, ::-1]
    return out


def decode_labels(mask):
    h, w = mask.shape
    mask[mask == 255] = 19
    color_table = np.array(label_colours, dtype=np.float32)
    out = np.take(color_table, mask, axis=0)
    out = out.astype(np.uint8)
    out = out[:, :, ::-1]
    return out
