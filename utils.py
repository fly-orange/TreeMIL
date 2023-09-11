import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
import torch.nn.functional as F

class PyraMIL():
    def __init__(self):
        self.bce = nn.BCELoss(reduction='mean')

    def total_loss(self, scores, wlabel):
        depth = len(scores)
        loss=0
        
        for i in range(depth):
            wpred = torch.max(scores[i], dim=1)[0]

            loss += self.bce(wpred, wlabel)

        return loss/depth

    def weak_loss(self, scores, wlabel):
        
        wpred = scores[-1][:, 0]  # 只考虑粗标签
        loss = self.bce(wpred, wlabel)
        
        return loss


    def last_loss(self, scores, wlabel):
        loss = self.bce(scores, wlabel)
        return loss


def ranking_loss(nscore, ascore):
    # nscore (B, T), ascore (B, T)
    maxn = torch.mean(nscore,dim=1)
    maxa = torch.mean(ascore,dim=1)
    tmp = F.relu(1. - maxa + maxn) # (B, )
    loss = tmp.mean()
    
    return loss

def compute_single_case(pred,label):
    
    correct = torch.mul(pred, label)
    TP, T, P = torch.sum(correct, dim=1), torch.sum(label, dim=1), torch.sum(pred,dim=1) # (B,)

    recall, precision, IoU = 1, 1, TP / (T + P - TP)  
    
    precision, recall, f1 = torch.ones(len(TP)), torch.ones(len(TP)), torch.ones(len(TP)) 
    for i in range(len(TP)):
        if T[i] != 0: recall[i] = TP[i] / T[i]   # (B)
        if P[i] != 0: precision[i] = TP[i]  / P[i]  # (B)

        if recall[i]==0.0 or precision[i]==0.0:
            f1[i] = 0.0
        else:
            f1[i] = 2*(recall[i]*precision[i])/(recall[i]+precision[i]) # (B)
    
    return f1, IoU


def compute_wacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_dacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_auc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    return metrics.auc(fpr, tpr)

def compute_bestf1(score, label, return_threshold=False):
    if isinstance(score, torch.Tensor):
        score = score.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()

    indices = np.argsort(score)[::-1]
    sorted_score = score[indices]
    sorted_label = label[indices]
    true_indices = np.where(sorted_label == 1)[0]

    bestf1 = 0.0
    best_threshold=None
    T = sum(label)
    for _TP, _P in enumerate(true_indices):
        TP, P = _TP + 1, _P + 1
        precision = TP / P
        recall = TP / T
        f1 = 2 * (precision*recall)/(precision+recall)
        threshold = sorted_score[_P] - np.finfo(float).eps
        if f1 > bestf1: # and threshold <= 0.5:
            bestf1 = f1
            best_threshold = sorted_score[_P] - np.finfo(float).eps
            #best_threshold = (sorted_score[_P-1] + sorted_score[_P]) / 2
    if return_threshold:
        return bestf1, best_threshold
    else:
        return bestf1

def compute_auprc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    return metrics.average_precision_score(label, pred)

def compute_precision_recall(result):
    TP, T, P = result
    recall, precision, IoU = 1, 1, TP / (T + P - TP)  
    if T != 0: recall = TP / T
    if P != 0: precision = TP / P

    if recall==0.0 or precision==0.0:
        f1 = 0.0
    else:
        f1 = 2*(recall*precision)/(recall+precision)
    return precision, recall, f1, IoU


'''Load params from the pretrained model'''

def load_model_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)

    return model


def build_finetune_model(model_new, model_pretrained):
    """
    Load pretrained weights to Pyramid model
    """
    # Load in pre-trained parameters
    model_new.embedding = model_pretrained.embedding
    model_new.conv_layers = model_pretrained.conv_layers
    model_new.layers = model_pretrained.layers

    return model_new