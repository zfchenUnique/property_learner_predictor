import json
import pdb
import torch
import numpy as np

def print_monitor(monitor, num_classes):
    acc_list =  []
    print('class accuracy: ')
    for c_id in range(num_classes):
        total = monitor['%d_count'%c_id]
        correct = monitor['%d_acc'%c_id]
        acc = correct / max(total, 0.000001)
        print('class: %d, acc: %1f'%(c_id, acc)) 
        acc_list.append(acc)
        print('predict distribution: ')
        pred_dist = [ele / max(total, 0.000001) for ele in monitor['%d_pred'%c_id] ]
        print(pred_dist)
    acc = np.mean(acc_list)
    print('class average acc: %1f\n'%(acc)) 
    return acc

def monitor_initialization(args):
    monitor = {}
    for cls_id in range(args.num_classes):
        monitor['%d_count'%cls_id] = 0.0
        monitor['%d_acc'%cls_id] = 0.0
        monitor['%d_pred'%cls_id] = []
        for c_id2 in range(args.num_classes):
            monitor['%d_pred'%cls_id].append(0.0)
    return monitor

def max_pool_prediction(output, num_obj, ref2query_list):
    pad_val = torch.min(output) - 1
    ref_num = len(ref2query_list)
    assert output.shape[0]==ref_num+1
    for ref_id in range(ref_num):
        visible_obj_list = list(ref2query_list[ref_id].values())
        mask_mat = np.zeros((num_obj, num_obj), dtype=np.float32)
        for id1 in range(num_obj):
            for id2 in range(num_obj):
                if (id1 not in visible_obj_list) or (id2 not in visible_obj_list):
                    mask_mat[id1, id2] = pad_val
                    mask_mat[id2, id1] = pad_val
        # Exclude self edges
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_obj, num_obj)) - np.eye(num_obj)),
            [num_obj, num_obj])
        mask_mat = np.reshape(mask_mat, -1)[off_diag_idx]
        mask_tensor = torch.from_numpy(mask_mat).to(output.device)
        output[ref_id+1] += mask_tensor.unsqueeze(dim=1)
    output, output_index = torch.max(output, dim=0, keepdim=True)
    return output

def compute_acc_by_class(output, target, num_classes, monitor):
    acc_list = []
    for c_id in range(num_classes):
        ele_idx = (target.view(-1)==c_id).nonzero().squeeze(dim=1)
        if ele_idx.shape[0]<=0:
            continue
        output_c = torch.index_select(output.view(-1, num_classes), 0, ele_idx)
        target_c =  torch.index_select(target.view(-1), 0, ele_idx)
        pred = output_c.data.max(1, keepdim=True)[1]
        correct = pred.eq(target_c.data.view_as(pred)).cpu().sum()
        monitor['%d_count'%c_id] += pred.size(0)
        monitor['%d_acc'%c_id] += correct
        acc = correct*1.0 / pred.size(0)
        acc_list.append(acc)
        for c_id2 in range(num_classes):
            monitor['%d_pred'%c_id][c_id2] += int(torch.sum(pred.view(-1)==c_id2))
    return monitor, acc_list
