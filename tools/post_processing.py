import json
import pdb
import os
import numpy as np

MASS_LIST = [1, 5]

def decode_square_edges(charge_pred, prp_num):
    charge_square = np.zeros((prp_num, prp_num, charge_pred.shape[1]))
    edge_id = 0
    for prp_id1 in range(prp_num):
        for prp_id2 in range(prp_num):
            if prp_id1==prp_id2:
                continue
            charge_square[prp_id1, prp_id2] = charge_pred[edge_id]
            edge_id +=1
    charge_square_trans = np.transpose(charge_square, (1, 0, 2))
    charge_out = 0.5 * (charge_square + charge_square_trans)
    return charge_out

def parse_results(prp_config_dir, des_prp_dir, raw_result_charge_path, raw_result_mass_path):
    with open(raw_result_charge_path, 'r') as fh:
        ann_charge = json.load(fh)['charge']
    with open(raw_result_mass_path, 'r') as fh:
        ann_mass = json.load(fh)['mass']
    if not os.path.isdir(des_prp_dir):
        os.makedirs(des_prp_dir)
    sim_str_list = sorted(list(ann_charge.keys()))
    for test_id, sim_str in enumerate(sim_str_list):
        src_config_full_path = os.path.join(prp_config_dir, sim_str +'.json') 
        des_config_full_path = os.path.join(des_prp_dir, sim_str +'.json') 
        with open(src_config_full_path, 'r') as  src_fh:
            src_config = json.load(src_fh)['config']
        prp_num = len(src_config)
        mass_pred = ann_mass[sim_str]
        # edge type: 0, 1, 2 for uncharged, same charge and opposite charge
        charge_pred = np.array(ann_charge[sim_str])
        assert prp_num == len(mass_pred)
        assert len(charge_pred) == prp_num * (prp_num - 1)
        square_edges = decode_square_edges(charge_pred, prp_num)
        edge_idx = np.argmax(square_edges, axis=2)        
        out_dict = {'config': [], 'edges':[]}
        config_list = []
        for obj_id, obj_config in enumerate(src_config): 
            tmp_obj = {}
            for attr in ['color', 'shape', 'material']:
                tmp_obj[attr] = obj_config[attr]
            tmp_obj['mass'] = 5 if mass_pred[obj_id]==1 else 1
            config_list.append(tmp_obj)            
        out_dict['config'] = config_list
        out_dict['edges'] = edge_idx.tolist()
        with open(des_config_full_path, 'w') as fh:
            json.dump(out_dict, fh)

if __name__=='__main__':
    raw_result_charge_path = 'logs/exp19/raw_prediction_charge.json'
    raw_result_mass_path = 'logs/exp19/raw_prediction_charge.json'
    prp_config_dir = '/home/zfchen/code/output/ns-vqa_output/v14_prp/config'
    des_prp_dir = '/home/zfchen/code/output/ns-vqa_output/v14_prp_pred/config'
    parse_results(prp_config_dir, des_prp_dir, raw_result_charge_path, raw_result_mass_path)
