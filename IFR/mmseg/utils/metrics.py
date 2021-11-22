import numpy as np
from sklearn.metrics import confusion_matrix


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        miou = np.nanmean(iou)
        if return_class:
            return miou, iou
        else:
            return miou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore_recall(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        iou = np.diag(hist) / hist.sum(axis=0)
        miou = np.nanmean(iou)
        if return_class:
            return miou, iou
        else:
            return miou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore_precision(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        iou = np.diag(hist) / hist.sum(axis=1)
        miou = np.nanmean(iou)
        if return_class:
            return miou, iou
        else:
            return miou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore_PA(object):
    def __init__(self):
        self.correct = 0.0
        self.total = 0.0

    def update(self, label, predict):
        mask = (label != 255)
        self.correct += np.sum(label == predict)
        self.total += np.sum(mask)

    def get_scores(self):
        pa = self.correct / self.total
        return pa

    def reset(self):
        self.correct = 0.0
        self.total = 0


class runningScore_CA(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.correct = [0.0 for i in range(n_classes)]
        self.total = [0.0 for i in range(n_classes)]
        self.ca_list = [0.0 for i in range(n_classes)]
        self.ca = 0.0

    def update(self, label, predict):
        for i in range(self.n_classes):
            self.total[i] += np.sum(label == i)
            self.correct[i] += np.sum((label == i) * (label == predict))

    def get_scores(self):
        for i in range(self.n_classes):
            acc = self.correct[i] / self.total[i]
            self.ca_list[i] = acc
            self.ca += acc
        self.ca /= self.n_classes

        return self.ca_list, self.ca

    def reset(self):
        self.correct = [0.0 for i in range(self.n_classes)]
        self.total = [0.0 for i in range(self.n_classes)]
        self.ca_list = [0.0 for i in range(self.n_classes)]
        self.ca = 0.0
