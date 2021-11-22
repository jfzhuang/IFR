import os.path as osp
import tempfile
import random

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset, CustomTemporalDataset, CustomFixMatchDataset, CustomTemporalFixMatchDataset
from .pipelines import Compose
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class CityscapesDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = (
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic light',
        'traffic sign',
        'vegetation',
        'terrain',
        'sky',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    )

    PALETTE = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(self, **kwargs):
        super(CityscapesDataset, self).__init__(
            img_suffix='_leftImg8bit.png', seg_map_suffix='_gtFine_labelTrainIds.png', **kwargs
        )

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels

        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels

            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: ' f'{len(results)} != {len(self)}'
        )

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir

    def evaluate(self, results, metric='mIoU', logger=None, imgfile_prefix=None, efficient_test=False):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            eval_results.update(super(CityscapesDataset, self).evaluate(results, metrics, logger, efficient_test))

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to ' 'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_files, tmp_dir = self.format_results(results, imgfile_prefix)

        if tmp_dir is None:
            result_dir = imgfile_prefix
        else:
            result_dir = tmp_dir.name

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = False
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = False
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in mmcv.scandir(self.ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results


@DATASETS.register_module()
class CityscapesSemiDataset(CustomDataset):
    CLASSES = (
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic light',
        'traffic sign',
        'vegetation',
        'terrain',
        'sky',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    )

    PALETTE = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(
        self,
        pipeline,
        img_dir,
        ann_dir,
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_gtFine_labelTrainIds.png',
        split=None,
        split_unlabeled=None,
        data_root=None,
        ignore_index=255,
        reduce_zero_label=False,
        classes=None,
        palette=None,
        clip_length=30,
        idx_sup=19,
    ):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.split_unlabeled = split_unlabeled
        self.data_root = data_root
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.clip_length = clip_length
        self.idx_sup = idx_sup

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)
            if not (self.split_unlabeled is None or osp.isabs(self.split_unlabeled)):
                self.split_unlabeled = osp.join(self.data_root, self.split_unlabeled)

        # load annotations
        self.video_infos_labeled, self.video_infos_unlabeled = self.load_annotations(
            self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split, self.split_unlabeled
        )

    def __len__(self):
        """Total number of samples of data."""
        return len(self.video_infos_labeled)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split, split_unlabeled):
        video_infos_labeled = []
        with open(split) as f:
            lines = f.readlines()

        for i in range(len(lines) // self.clip_length):
            video_lines = lines[i * self.clip_length : (i + 1) * self.clip_length]
            img_infos = []
            for line in video_lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix, ann=img_name + seg_map_suffix)
                img_infos.append(img_info)
            video_infos_labeled.append(img_infos)

        print_log(f'Loaded {len(video_infos_labeled)} labeled clips', logger=get_root_logger())

        video_infos_unlabeled = []
        with open(split_unlabeled) as f:
            lines = f.readlines()

        for i in range(len(lines) // self.clip_length):
            video_lines = lines[i * self.clip_length : (i + 1) * self.clip_length]
            img_infos = []
            for line in video_lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix, ann=img_name + seg_map_suffix)
                img_infos.append(img_info)
            video_infos_unlabeled.append(img_infos)

        print_log(f'Loaded {len(video_infos_unlabeled)} unlabeled clips', logger=get_root_logger())
        return video_infos_labeled, video_infos_unlabeled

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir

    def __getitem__(self, video_idx_0):
        idx_v0_0 = self.idx_sup
        idx_v0_1 = random.choice([i for i in range(self.clip_length) if i != idx_v0_0])

        video_idx_1 = random.randint(0, len(self.video_infos_unlabeled) - 1)
        idx_list = [i for i in range(self.clip_length)]
        random.shuffle(idx_list)
        idx_v1_0, idx_v1_1 = idx_list[:2]

        return self.prepare_train_img(video_idx_0, idx_v0_0, idx_v0_1, video_idx_1, idx_v1_0, idx_v1_1)

    def prepare_train_img(self, video_idx_0, idx_v0_0, idx_v0_1, video_idx_1, idx_v1_0, idx_v1_1):
        img_info_v0_0 = self.video_infos_labeled[video_idx_0][idx_v0_0]
        img_info_v0_1 = self.video_infos_labeled[video_idx_0][idx_v0_1]
        img_info_v1_0 = self.video_infos_unlabeled[video_idx_1][idx_v1_0]
        img_info_v1_1 = self.video_infos_unlabeled[video_idx_1][idx_v1_1]
        results = dict(
            img_info_v0_0=img_info_v0_0,
            img_info_v0_1=img_info_v0_1,
            img_info_v1_0=img_info_v1_0,
            img_info_v1_1=img_info_v1_1,
        )
        self.pre_pipeline(results)
        return self.pipeline(results)
