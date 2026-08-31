#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:02:47 2025

@author: sowmyabalagala
"""

# metrics.py

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score

class AUCWithMaskedClass:
    def __init__(self):
        self.scores = []
        self.labels = []

    def update_state(self, y_true, y_pred):
        self.labels.extend(y_true.flatten().tolist())
        self.scores.extend(y_pred.flatten().tolist())

    def result(self):
        try:
            return torch.tensor(roc_auc_score(self.labels, self.scores))
        except:
            return torch.tensor(0.0)


class PrecisionWithMaskedClass:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def update_state(self, y_true, y_pred):
        self.y_true.extend(y_true.flatten().tolist())
        self.y_pred.extend(y_pred.flatten().tolist())

    def result(self):
        try:
            return torch.tensor(precision_score(self.y_true, self.y_pred, zero_division=0))
        except:
            return torch.tensor(0.0)


class RecallWithMaskedClass:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def update_state(self, y_true, y_pred):
        self.y_true.extend(y_true.flatten().tolist())
        self.y_pred.extend(y_pred.flatten().tolist())

    def result(self):
        try:
            return torch.tensor(recall_score(self.y_true, self.y_pred, zero_division=0))
        except:
            return torch.tensor(0.0)


def masked_iou(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / (union + 1e-8)


def compute_best_threshold(model, dataset, steps=100, thresholds=np.linspace(0.1, 0.9, 17)):
    best_f1 = -1
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for threshold in thresholds:
        precision_metric = PrecisionWithMaskedClass()
        recall_metric = RecallWithMaskedClass()

        model.eval()
        with torch.no_grad():
            for i, (feats_tf, labs_tf) in enumerate(dataset):
                if i >= steps:
                    break
                x = torch.from_numpy(feats_tf.numpy()).permute(0, 3, 1, 2).float().cuda()
                y_pred = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
                y_true = labs_tf.numpy()[..., 0]
                pred_bin = (y_pred > threshold).astype(np.float32)
                precision_metric.update_state(torch.tensor(y_true), torch.tensor(pred_bin))
                recall_metric.update_state(torch.tensor(y_true), torch.tensor(pred_bin))

        prec = precision_metric.result().item()
        rec = recall_metric.result().item()
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = prec
            best_recall = rec

    return best_threshold, best_f1, best_precision, best_recall
