import json
import pdb
import torch
import numpy as np

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
