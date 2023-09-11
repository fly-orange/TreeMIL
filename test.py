import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from pickle import dump

import os, time
import argparse
import copy

from model import Pyramid
from timeseries import TimeSeriesWithAnomalies, NormalTimeSeries, AbnormalTimeSeries
from softdtw_cuda import SoftDTW
from utils import * 

def evaluation(args, model, data_loader, pyra, train_loader=None):
    model.eval()
    total, total_loss, total_bce_loss = 0, 0, 0
    wscores, wlabels = [], []
    dscores, dlabels = [], []
    f1_result, iou_result = [], []
    wresult, dresult = np.zeros(3), np.zeros(3)

    for itr, batch in enumerate(data_loader):
        data = batch['data'].cuda()
        wlabel = batch['wlabel'].cuda()
        dlabel = batch['dlabel'].cuda()
        
        with torch.no_grad():
            out = model.get_scores(data)
            bce_loss = pyra.last_loss(out['wscore'], wlabel)
            # dtw_loss = model.dtw_loss(out['output'], wlabel).mean(0)
            loss = bce_loss

            total += data.size(0)
            total_bce_loss += bce_loss.item() * data.size(0)
            # total_dtw_loss += dtw_loss.item() * data.size(0)
            total_loss += loss.item() * data.size(0)

            # weak prediction 使用get_score函数
            wresult += compute_wacc(out['wpred'], wlabel)
            wscores.append(out['wscore'])
            wlabels.append(wlabel)

            # dense prediction 使用get_dpred函数
            dpred = model.get_dpred(out['output'], out['wpred']) # dense决策依赖于weak决策
            dresult += compute_dacc(dpred, dlabel)
            dscores.append(out['dscore'])
            dlabels.append(dlabel)
            case = compute_single_case(dpred, dlabel)
            # print(case[0][wlabel.detach().cpu().numpy()>0.5])
            f1_result.append(case[0][wlabel.detach().cpu().numpy()>0.5])
            iou_result.append(case[1][wlabel.detach().cpu().numpy()>0.5])
    
    f1_result = torch.cat(f1_result, dim=0)
    iou_result = torch.cat(iou_result, dim=0) 


    if train_loader is not None:
        for itr, batch in enumerate(train_loader):
            data = batch['data'].cuda()
            wlabel = batch['wlabel'].cuda()
            
            with torch.no_grad():
                out = model.get_scores(data)
                wscores.append(out['wscore'])
                wlabels.append(wlabel)

    ret = {}
    ret['loss'] = total_loss / total
    ret['bce_loss'] = total_bce_loss / total
    # ret['dtw_loss'] = total_dtw_loss / total
    

    # Weak and dense results under predefined threshold
    ret['wprecision'], ret['wrecall'], ret['wf1'], ret['wIoU'] = compute_precision_recall(wresult)
    ret['dprecision'], ret['drecall'], ret['df1'], ret['dIoU'] = compute_precision_recall(dresult)

    wscores, wlabels = torch.cat(wscores, dim=0), torch.cat(wlabels, dim=0)
    dscores, dlabels = torch.cat(dscores, dim=0), torch.cat(dlabels, dim=0)

    # Weak Result Curve and best
    ret['wauc'] = compute_auc(wscores, wlabels)
    ret['wauprc'] = compute_auprc(wscores, wlabels)
    ret['wbestf1'], ret['global_threshold'] = compute_bestf1(wscores, wlabels, return_threshold=True)
    wbestpred = (wscores >= ret['global_threshold']).type(torch.cuda.FloatTensor)
    wbestresult = compute_dacc(wbestpred, wlabels)
    ret['wbprecision'], ret['wbrecall'], ret['wbf1'], ret['wbIoU'] = compute_precision_recall(wbestresult)

    # Dense Result Curve and best
    ret['dauc'] = compute_auc(dscores, dlabels)
    ret['dauprc'] = compute_auprc(dscores, dlabels)
    ret['dbestf1'], ret['local_threshold'] = compute_bestf1(dscores, dlabels, return_threshold=True)
    dbestpred = (dscores >= ret['local_threshold']).type(torch.cuda.FloatTensor)
    dbestresult = compute_dacc(dbestpred, dlabels)
    ret['dbprecision'], ret['dbrecall'], ret['dbf1'], ret['dbIoU'] = compute_precision_recall(wbestresult)
    return ret, f1_result, iou_result


def test(args, train_dataset, valid_dataset, test_dataset):

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtw = SoftDTW(use_cuda=True, gamma=args.gamma, normalize=False)

    model = Pyramid(input_size=train_dataset.input_size, seq_size=args.split_size, ary_size=args.ary_size, inner_size=args.inner_size,
                    d_model=args.d_model, n_layer=args.n_layer,  n_head=args.n_head, d_k=args.d_k, d_v=args.d_v, d_inner_hid=args.d_inner_hid, agg_type=args.agg_type, dropout=args.dropout,
                    pooling_type='max', granularity=1, local_threshold=0.5, global_threshold=0.5, beta=10, dtw=dtw)
    model = torch.load(f'result/{args.dataset}/model_{args.agg_type}_{args.pooling_type}.pkl')
    print("============================")
    print("  Pretrained model loaded  ")
    print("============================")


    model.cuda()
    
    pyra = PyraMIL()

    # Start Testing

    valid_result, _, _ = evaluation(args, model, valid_loader, pyra)
    model.global_threshold = valid_result['global_threshold']
    test_result, f1_result, iou_result = evaluation(args, model, test_loader, pyra)

    total_results = [f1_result.detach().cpu().numpy(), iou_result.detach().cpu().numpy()]
    with open(f'result/{args.dataset}/results_{args.agg_type}_{args.pooling_type}.npy','wb') as f:
        dump(total_results, f)


    print("============================")
    print("  Final evaluation results  ")
    print("============================")

    print('\tTest  (WEAK) AUC : {:.6f}, AUPRC : {:.6f}, Best F1 : {:.6f}, Precision : {:.6f}, Recall : {:.6f}'.format(
                test_result['wauc'], test_result['wauprc'], test_result['wbestf1'], test_result['wprecision'], test_result['wrecall']))
    print('\tTest (DENSE) F1 : {:.6f}, Precision : {:.6f}, Recall : {:.6f}, IoU : {:.6f}'.format(
        test_result['df1'], test_result['dprecision'], test_result['drecall'], test_result['dIoU']))

def main(args):

    train_dataset = TimeSeriesWithAnomalies(args.data_dir, args.split_size, 'train')
    valid_dataset = TimeSeriesWithAnomalies(args.data_dir, args.split_size, 'valid')
    test_dataset = TimeSeriesWithAnomalies(args.data_dir, args.split_size, 'test')


    test(args, train_dataset, valid_dataset, test_dataset)

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser()

    # for C-ary tree Framework
    parser.add_argument('--ary_size', default=2, type=int, help='N-ary tree')
    parser.add_argument('--inner_size', default=3, type=int, help='adjacent node used in the same layer')
    parser.add_argument('--pooling_type', default='max', type=str, help='avg | max')
    parser.add_argument('--local_threshold', default=0.3, type=float, help='score threshold to identify anomalies')
    parser.add_argument('--granularity', default=4, type=int, help='granularity for sequential pseudo-labels')  # default: 4
    parser.add_argument('--beta', default=0.1, type=float, help='margin size for the alignment loss')
    parser.add_argument('--gamma', default=0.1, type=float, help='smoothing for differentiable DTW')


    # for Pyramid Net
    parser.add_argument('--d_model', default=64, type=int, help='hidden dimension for attn')
    parser.add_argument('--d_k', default=64, type=int, help='key dimension')
    parser.add_argument('--d_v', default=64, type=int, help='value dimension')
    parser.add_argument('--d_inner_hid', default=32, type=int, help='hidden dimension for ffn')
    parser.add_argument('--n_head', default=5, type=int, help='number of attention head')
    parser.add_argument('--n_layer', default=2, type=int, help='# of layers in the dicnn')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')

    # for Pretrain
    parser.add_argument('--fine_tune', default=False, action='store_true', help='whether to fine tune the pretrained model')
    
    
    # for Optimization
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--n_epochs', default=200, type=int, help="# of training epochs")
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--gpuidx', default=0, type=int, help='gpu index')
    parser.add_argument('--patience', default=50, type=int, help='# of patience for early stopping')
    parser.add_argument('--stopping', default='f1', type=str, help='f1 | loss')
    parser.add_argument('--agg_type', default='max', type=str, help='avg | max | conv')
    parser.add_argument('--seed', default=0, type=int)
    
    # for Dataset
    parser.add_argument('--dataset', default='CC', type=str, help='EMG | GHL | SMD | SMAP | PSM | MSL')
    parser.add_argument('--split_size', default=120, type=int, help='split size for preprocessing the data')

    args = parser.parse_args()

    # GPU setting
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)

    # Random seed initialization
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    random.seed(args.seed)

    args.data_dir = './data/' + args.dataset
    args.load_model_path = f'result/{args.dataset}/pretrain_model.pkl'

    print(args)
    main(args=args)
