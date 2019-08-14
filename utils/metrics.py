import numpy as np


class Evaluator:
    def __init__(self, num_classes):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.dc_per_case = np.zeros(self.num_classes)
        self.dc_each_case = []
        self.num_case = np.zeros(self.num_classes)
    
    def _generate_matrix(self, pred, label):
        mask = (label >= 0) & (label < self.num_classes)
        label = self.num_classes * label[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix
    
    def dice_coef(self, class_, confusion_matrix):
        dc = confusion_matrix[class_][class_] * 2 / (
                np.sum(confusion_matrix, axis=0)[class_] + np.sum(confusion_matrix, axis=1)[class_])
        if np.isnan(dc):
            dc = -1
        return dc
    
    def add_batch(self, preds, labels):
        assert preds.shape == labels.shape
        for i in range(len(preds)):
            self.add(preds[i], labels[i])
    
    def add(self, pred, label):
        assert pred.shape == label.shape
        matrix = self._generate_matrix(pred, label)
        self.confusion_matrix += matrix
        
        dc_case = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            dc = self.dice_coef(cls, matrix)
            if dc != -1:
                self.dc_per_case[cls] += dc
                self.num_case[cls] += 1
            dc_case[cls] = dc
        self.dc_each_case.append(dc_case)
    
    def eval(self):
        acc = dict()
        for cls in range(self.num_classes):
            dc_per_case = self.dc_per_case[cls] / self.num_case[cls]
            dc_global = self.dice_coef(cls, self.confusion_matrix)
            acc[f'dc_per_case_{cls}'] = dc_per_case
            acc[f'dc_global_{cls}'] = dc_global
        acc[f'dc_each_case'] = self.dc_each_case
        return acc
