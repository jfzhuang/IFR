import os
import cv2
import glob


def main_seq():
    gt_path = './data/cityscapes/leftImg8bit_sequence/train'
    save_path = './data/cityscapes/leftImg8bit_sequence_down_2x/train'

    subdirs = sorted(glob.glob(os.path.join(gt_path, '*')))
    for i, subdir in enumerate(subdirs):
        subdir = subdir.replace(gt_path + '/', '')
        names = sorted(glob.glob(os.path.join(gt_path, subdir, '*.png')))
        if not os.path.exists(os.path.join(save_path, subdir)):
            os.makedirs(os.path.join(save_path, subdir))
        for j, name in enumerate(names):
            print('{}/{} {}/{}'.format(i, len(subdirs), j, len(names)))
            name = name.replace(os.path.join(gt_path, subdir) + '/', '')
            gt = cv2.imread(os.path.join(gt_path, subdir, name))
            gt = cv2.resize(gt, (1024, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_path, subdir, name), gt)

    gt_path = './data/cityscapes/leftImg8bit_sequence/val'
    save_path = './data/cityscapes/leftImg8bit_sequence_down_2x/val'

    subdirs = sorted(glob.glob(os.path.join(gt_path, '*')))
    for i, subdir in enumerate(subdirs):
        subdir = subdir.replace(gt_path + '/', '')
        names = sorted(glob.glob(os.path.join(gt_path, subdir, '*.png')))
        if not os.path.exists(os.path.join(save_path, subdir)):
            os.makedirs(os.path.join(save_path, subdir))
        for j, name in enumerate(names):
            print('{}/{} {}/{}'.format(i, len(subdirs), j, len(names)))
            name = name.replace(os.path.join(gt_path, subdir) + '/', '')
            gt = cv2.imread(os.path.join(gt_path, subdir, name))
            gt = cv2.resize(gt, (1024, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_path, subdir, name), gt)


def main_gt():
    gt_path = './data/cityscapes/gtFine/train'
    save_path = './data/cityscapes/gtFine_down_2x/train'

    subdirs = sorted(glob.glob(os.path.join(gt_path, '*')))
    for i, subdir in enumerate(subdirs):
        subdir = subdir.replace(gt_path + '/', '')
        names = sorted(glob.glob(os.path.join(gt_path, subdir, '*_gtFine_labelTrainIds.png')))
        if not os.path.exists(os.path.join(save_path, subdir)):
            os.makedirs(os.path.join(save_path, subdir))
        for j, name in enumerate(names):
            print(i, j)
            name = name.replace(os.path.join(gt_path, subdir) + '/', '')
            gt = cv2.imread(os.path.join(gt_path, subdir, name), 0)
            gt = cv2.resize(gt, (1024, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(save_path, subdir, name), gt)

    gt_path = './data/cityscapes/gtFine/val'
    save_path = './data/cityscapes/gtFine_down_2x/val'

    subdirs = sorted(glob.glob(os.path.join(gt_path, '*')))
    for i, subdir in enumerate(subdirs):
        subdir = subdir.replace(gt_path + '/', '')
        names = sorted(glob.glob(os.path.join(gt_path, subdir, '*_gtFine_labelTrainIds.png')))
        if not os.path.exists(os.path.join(save_path, subdir)):
            os.makedirs(os.path.join(save_path, subdir))
        for j, name in enumerate(names):
            print(i, j)
            name = name.replace(os.path.join(gt_path, subdir) + '/', '')
            gt = cv2.imread(os.path.join(gt_path, subdir, name), 0)
            gt = cv2.resize(gt, (1024, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(save_path, subdir, name), gt)


if __name__ == '__main__':
    main_seq()
    main_gt()
