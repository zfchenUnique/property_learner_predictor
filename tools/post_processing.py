import json
import pdb
import os
import numpy as np
import argparse

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

def check_consistent(obj_id, out_dict, assign_objs):
    pdb.set_trace()

def assigning_charge_to_objects(out_dict, square_edges):
    """
    Assign the objects that have the highest confidence with charges
    Remove charge edges if conflicts with the assigned objects
    """
    edge_idx = np.argmax(square_edges, axis=2)        
    edge_score = np.max(square_edges, axis=2)
    valid_loc = np.where(edge_idx>0)
    assign_objs = []
    assign_edges = np.zeros(edge_idx.shape)
    while True:
        max_val = np.amax(edge_score)
        edge_loc = np.where(edge_score==max_val) 
        out_break_flag = False
        #print('max_val: %f\n'%(max_val))
        for idx in range(len(edge_loc[0])):
            obj1= edge_loc[0][idx]
            obj2= edge_loc[1][idx]
            #print('obj1: %d, obj2: %d\n'%(obj1, obj2))
            edge_score[obj1, obj2] = np.amin(edge_score) - 1 
            edge_score[obj2, obj1] =  edge_score[obj1, obj2]
            assign_edges[obj1, obj2] = 1
            assign_edges[obj2, obj1] = 1
            if edge_idx[obj1, obj2]==0:
                #out_dict['config'][obj1]['charge']=0
                #out_dict['config'][obj2]['charge']=0
                #assign_objs.append(obj1)
                #assign_objs.append(obj2)
                continue

            if obj1 not in assign_objs:
                if obj2 not in assign_objs:
                    out_dict['config'][obj1]['charge']=1
                elif out_dict['config'][obj2]['charge']==0:
                    out_dict['config'][obj1]['charge']==0
                elif edge_idx[obj1, obj2]==2:
                    out_dict['config'][obj1]['charge']= out_dict['config'][obj2]['charge']*-1
                elif edge_idx[obj1, obj2]==1:
                    out_dict['config'][obj1]['charge']= out_dict['config'][obj2]['charge']
                else:
                    out_break_flag = True
                    break
                assign_objs.append(obj1)
            if obj2 not in assign_objs:
                if edge_idx[obj1, obj2] ==1:
                    out_dict['config'][obj2]['charge']= out_dict['config'][obj1]['charge']
                elif out_dict['config'][obj1]['charge']==0:
                    out_dict['config'][obj2]['charge']==0
                elif edge_idx[obj1, obj2] ==2:
                    out_dict['config'][obj2]['charge']= out_dict['config'][obj1]['charge']*-1
                else:
                    out_break_flag = True
                    break
                assign_objs.append(obj2)
        if len(assign_objs)>=len(out_dict['config']):
            break
        obj_num = len(out_dict['config'])
        #print(np.sum(assign_edges))
        if np.sum(assign_edges)== obj_num * obj_num:
            break
        if out_break_flag:
            break
    for obj_id, obj_info in enumerate(out_dict['config']):
        if 'charge' not in obj_info:
            out_dict['config'][obj_id]['charge'] = 0

def parse_results(prp_config_dir, des_prp_dir, raw_result_charge_path, raw_result_mass_path, gt_flag):
    with open(raw_result_charge_path, 'r') as fh:
        ann_charge = json.load(fh)['charge']
    with open(raw_result_mass_path, 'r') as fh:
        ann_mass = json.load(fh)['mass']
    if not os.path.isdir(des_prp_dir):
        os.makedirs(des_prp_dir)
    sim_str_list = sorted(list(ann_charge.keys()))
    for test_id, sim_str in enumerate(sim_str_list):
        if not gt_flag: 
            src_config_full_path = os.path.join(prp_config_dir, sim_str +'.json') 
        else:
            src_config_full_path = os.path.join(prp_config_dir, sim_str, 'annotations', 'annotation.json')
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
        out_dict['edges'] = square_edges.tolist()
        assigning_charge_to_objects(out_dict, square_edges)
        #pdb.set_trace()
        with open(des_config_full_path, 'w') as fh:
            json.dump(out_dict, fh)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_result_charge_path', type=str, default='logs/exp19/raw_prediction_charge.json')
    parser.add_argument('--raw_result_mass_path', type=str, default='logs/exp19/raw_prediction_charge.json')
    parser.add_argument('--prp_config_dir', type=str, default='/home/zfchen/code/output/ns-vqa_output/v14_prp/config')
    parser.add_argument('--des_prp_dir', type=str, default='/home/zfchen/code/output/ns-vqa_output/v14_prp_pred_v2/config')
    parser.add_argument('--gt_flag', type=int, default=0)
    args = parser.parse_args()
    raw_result_charge_path = args.raw_result_charge_path
    raw_result_mass_path = args.raw_result_mass_path
    prp_config_dir = args.prp_config_dir
    des_prp_dir = args.des_prp_dir
    gt_flag = args.gt_flag
    parse_results(prp_config_dir, des_prp_dir, raw_result_charge_path, raw_result_mass_path, gt_flag)
