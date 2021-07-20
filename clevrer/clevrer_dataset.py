import argparse
import json
import numpy as np
import os
import pdb
from torch.utils.data import DataLoader, Dataset
import torch
import glob
import random

IMG_H = 320
IMG_W = 480

def check_in_field(pos, field_config):
    """
    Return flag to indicate whether the object is in the field
    """
    k1 = field_config['borderline'][0]
    k2 = field_config['borderline'][1]
    b  = field_config['borderline'][2]
    flag = k1 * pos[0] + k2 * pos[1] + b > 0
    return flag

def get_field_info_sim(track, ann, add_field_flag):
    field = np.zeros((track.shape[0], track.shape[1], 1 )).astype(np.float32)
    if len(ann['field_config'])==0: 
        return field
    if not add_field_flag:
        return field
    border = np.array(ann['field_config'][0]['borderline'][:2]).reshape((1, 1, 2))
    field_flag = np.sum(track * border, axis=2)  + ann['field_config'][0]['borderline'][2]
    field = (field_flag >  0 ).astype(np.float32) 
    return field.reshape(track.shape[0], track.shape[1], 1)  

def get_one_hot_for_shape(shape_str):
    if shape_str=='sphere':
        return np.array([1, 0, 0])
    elif shape_str =='cylinder':
        return np.array([0, 1, 0])
    elif shape_str =='cube':
        return np.array([0, 0, 1])

def load_obj_track(track_path,  num_vis_frm, pad_value=-1):
    track_ori = np.load(track_path)
    track_ori = np.transpose(track_ori, [1, 0, 2])
    track_pad = track_ori
    obj_num, time_step, box_dim = track_ori.shape
    pad_pos = track_ori == np.array([0, 0, IMG_W, IMG_H])
    pad_obj_frm = np.sum(pad_pos, axis=2)==4 
    # coordinate normalization [x, y, w, h]
    track_pad[:, :, 0] /= IMG_W
    track_pad[:, :, 2] /= IMG_W
    track_pad[:, :, 1] /= IMG_H 
    track_pad[:, :, 3] /= IMG_H
    track_pad[:, :, 2] -= track_pad[:, :, 0]
    track_pad[:, :, 3] -= track_pad[:, :, 1]
    track_pad[pad_obj_frm] = pad_value 

    track = -1 * np.ones((obj_num, num_vis_frm, box_dim)) 
    frm_num = min(time_step, num_vis_frm)
    track[:, :frm_num] = track_pad[:, :frm_num]
    vel = np.zeros((obj_num, num_vis_frm, box_dim))
    vel[:,1:frm_num] = track[:,1:frm_num] - track[:, : frm_num-1]
    vel[:, 0] = vel[:, 1]
    return track, vel

def sample_obj_track(motion, num_vis_frm, sample_every):
    obj_track_list = []
    for idx, tmp_m in enumerate(motion):
        if idx % sample_every !=0:
            continue
        loc_list = []
        for obj_id, m_info in enumerate(tmp_m):
            loc = m_info['location']
            loc_list.append(loc)
        obj_track_list.append(loc_list)
        if len(obj_track_list) >= num_vis_frm:
            break
    track_ori = np.array(obj_track_list)
    track_ori = np.transpose(track_ori, [1, 0, 2])
    obj_num, time_step, loc_dim = track_ori.shape
    frm_num = min(time_step, num_vis_frm)
    track = -1 * np.ones((obj_num, num_vis_frm, loc_dim)) 
    track[:, :frm_num] = track_ori[:, :frm_num]
    vel = np.zeros((obj_num, num_vis_frm, loc_dim))
    vel[:,1:frm_num] = track[:,1:frm_num] - track[:, : frm_num-1]
    vel[:, 0] = vel[:, 1]
    track = track[:,:,:2]
    vel = vel[:, :, :2]
    #track = np.reshape(track, [obj_num, -1])
    #vel = np.reshape(vel, [obj_num, -1])
    return track, vel

def get_edge_rel(obj_list, visible_obj_list=None):
    """
    edge type: 0, 1, 2 for uncharged, same charge and opposite charge
    edge size: num_obj *(num_obj-1) 
    """
    num_obj = len(obj_list)
    edge = np.zeros((num_obj, num_obj ))
    charge_id_list = [ obj_id for obj_id, obj_info in enumerate(obj_list) if 'charge' in obj_info and obj_info['charge']!=0]
    for id1 in charge_id_list:
        for id2 in charge_id_list:
            if id1==id2:
                continue
            if visible_obj_list is not None:
                if (id1 not in visible_obj_list) or (id2 not in visible_obj_list):
                    continue
            if obj_list[id1]['charge']==obj_list[id2]['charge']:
                edge[id1, id2] = 1
                edge[id2, id1] = 1
            else:
                edge[id1, id2] = 2
                edge[id2, id1] = 2
    num_atoms = len(obj_list)
    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edge = np.reshape(edge, -1)[off_diag_idx]
    return edge

class clevrerDataset(Dataset):
    def __init__(self, args, sim_st_idx, sim_ed_idx, split, data_aug_flag=False, ref_aug_flag=False, mask_aug_prob=0):
        self.ref_num = args.ref_num
        self.num_vis_frm = args.num_vis_frm
        #self.sim_st_idx = sim_st_idx
        #self.sim_ed_idx = sim_ed_idx
        self.sim_list = list(range(sim_st_idx, sim_ed_idx))
        # adding extra training idx
        if split=='train':
            sup_list = list(range(args.train_st_idx2, args.train_ed_idx2))
            new_train_list = sorted(list(set(self.sim_list + sup_list )))
            self.sim_list = new_train_list

        self.data_aug_flag = data_aug_flag
        self.ref_aug_flag = ref_aug_flag
        self.mask_aug_prob = mask_aug_prob
        self.args = args
        self.split = split
        if split=='val':
            self.ann_dir = args.ann_dir_val
            self.track_dir = args.track_dir_val
            self.ref_dir = args.ref_dir_val
            self.ref_track_dir = args.ref_track_dir_val
        elif split=='train' or split=='test':
            self.ann_dir = args.ann_dir
            self.track_dir = args.track_dir
            self.ref_dir = args.ref_dir
            self.ref_track_dir = args.ref_track_dir

    def __len__(self):
        #return self.sim_ed_idx - self.sim_st_idx 
        return len(self.sim_list)

    def __getitem__(self, index):
        if self.args.sim_data_flag:
            return self.__getitem_sim__(index)
        else:
            return self.__getitem_render__(index)

    def __getitem_sim__(self, index):
        """
        obj_ftr: concatenate ( shape_one_hot, loc, vel )
        edge: of shape (num_obj * (num_obj-1))
        """
        sim_id = self.sim_list[index] 
        sim_str = 'sim_%05d'%(sim_id)
        ann_path = os.path.join(self.ann_dir, sim_str + '.json')
        with open(ann_path, 'r') as fh:
            ann = json.load(fh)
        # shape info
        shape_emb = [ get_one_hot_for_shape(obj_info['shape']) for obj_info in ann['config']]
        shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
        shape_mat_exp = np.repeat(shape_mat, self.args.num_vis_frm, axis=1)
        track, vel = sample_obj_track(ann['motion'], self.args.num_vis_frm, self.args.sample_every)
        # mass info
        mass_list = [obj_info['mass']==5 for obj_info in ann['config']]
        mass_label = np.array(mass_list).astype(np.long)
        mass_label = torch.from_numpy(mass_label)
        # field info
        field = get_field_info_sim(track, ann, self.args.add_field_flag)
        
        obj_ftr = np.concatenate([shape_mat_exp, track, vel, field], axis=2)
        # edge info
        edge = get_edge_rel(ann['config'])
        obj_ftr = obj_ftr.astype(np.float32)
        edge = edge.astype(np.long)
        obj_ftr = torch.from_numpy(obj_ftr)
        edge = torch.from_numpy(edge)

        if self.args.load_reference_flag: 
            obj_ftr_list, edge_list, ref2query_list = load_reference_ftr_sim(self.ref_dir, sim_str, ann, self.args)
            if self.ref_aug_flag:
                ref_num= len(obj_ftr_list) 
                if ref_num==0:
                    print(ann_path)
                smp_num = random.randint(1, ref_num)
                smp_id_list = random.sample(list(range(ref_num)), smp_num)
                obj_ftr_list = [obj_ftr_list[smp_id] for smp_id in smp_id_list]
                edge_list = [edge_list[smp_id] for smp_id in smp_id_list]
                ref2query_list = [ref2query_list[smp_id] for smp_id in smp_id_list]
            obj_ftr_list.insert(0, obj_ftr)
            edge_list.insert(0, edge)
            obj_ftr = torch.stack(obj_ftr_list, dim=0)
            edge = torch.stack(edge_list, dim=0)
        else:
            ref2query_list = []
            obj_ftr = obj_ftr.unsqueeze(dim=0)
            edge = edge.unsqueeze(dim=0)
        valid_flag = None
        return obj_ftr, edge, ref2query_list, sim_str, mass_label, valid_flag 

    def __getitem_render__(self, index):
        """
        obj_ftr: concatenate ( shape_one_hot, loc, vel )
        edge: of shape (num_obj * (num_obj-1))
        """
        sim_id = self.sim_list[index]
        sim_str = 'sim_%05d'%(sim_id)
        if self.args.proposal_flag:
            ann_path = os.path.join(self.ann_dir, sim_str + '.json')
        else:
            ann_path = os.path.join(self.ann_dir, sim_str, 'annotations', 'annotation.json')
        with open(ann_path, 'r') as fh:
            ann = json.load(fh)
        shape_emb = [ get_one_hot_for_shape(obj_info['shape']) for obj_info in ann['config']]
        shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
        shape_mat_exp = np.repeat(shape_mat, self.args.num_vis_frm, axis=1)
        # mass info
        mass_list = [ 'mass' in obj_info and obj_info['mass']==5 for obj_info in ann['config']]
        mass_label = np.array(mass_list).astype(np.long)
        mass_label = torch.from_numpy(mass_label)
        
        # load object track
        track_path = os.path.join(self.track_dir, sim_str+'.npy')
        track, vel = load_obj_track(track_path, self.num_vis_frm)
        obj_ftr = np.concatenate([shape_mat_exp, track, vel], axis=2)
        edge = get_edge_rel(ann['config'])
        obj_ftr = obj_ftr.astype(np.float32)
        edge = edge.astype(np.long)
        obj_ftr = torch.from_numpy(obj_ftr)
        edge = torch.from_numpy(edge)
        if self.args.load_reference_flag: 
            ref_dir = os.path.join(self.ref_dir, sim_str)
            obj_ftr_list, edge_list, ref2query_list = load_reference_ftr(ref_dir, self.ref_track_dir, sim_str, ann, self.args)
            if self.ref_aug_flag:
                ref_num= len(obj_ftr_list) 
                smp_num = random.randint(1, ref_num)
                smp_id_list = random.sample(list(range(ref_num)), smp_num)
                obj_ftr_list = [obj_ftr_list[smp_id] for smp_id in smp_id_list]
                edge_list = [edge_list[smp_id] for smp_id in smp_id_list]
                ref2query_list = [ref2query_list[smp_id] for smp_id in smp_id_list]
            obj_ftr_list.insert(0, obj_ftr)
            edge_list.insert(0, edge)
            obj_ftr = torch.stack(obj_ftr_list, dim=0)
            edge = torch.stack(edge_list, dim=0)
        else:
            ref2query_list = []
            obj_ftr = obj_ftr.unsqueeze(dim=0)
            edge = edge.unsqueeze(dim=0)
        # valid masks for object tracks 
        valid_flag1 = (obj_ftr[:, :, :, 3] >0).type(torch.uint8) 
        valid_flag2 = (obj_ftr[:, :, :, 3] <1).type(torch.uint8) 
        valid_flag3 = (obj_ftr[:, :, :, 4] >0).type(torch.uint8) 
        valid_flag4 = (obj_ftr[:, :, :, 4] <1).type(torch.uint8) 
        if self.data_aug_flag:
            obj_ftr[:, :, :, 3:] = obj_ftr[:, :, :, 3:] + torch.randn(obj_ftr[:, :, :, 3:].size()) * self.args.data_noise_weight
        if np.random.rand() < self.mask_aug_prob:
            tmp_mask = torch.randn(obj_ftr[:, :, :, 3:].shape) > 0.1 
            tmp_mask = tmp_mask.type(torch.FloatTensor)
            obj_ftr[:, :, :, 3:] = obj_ftr[:, :, :, 3:] * tmp_mask
        valid_flag  = valid_flag1 +  valid_flag2  + valid_flag3 + valid_flag4 ==4
        return obj_ftr, edge, ref2query_list, sim_str, mass_label, valid_flag 

def map_ref_to_query(obj_list_query, obj_list_ref):
    ref2query ={}
    for idx1, obj_info1 in enumerate(obj_list_ref):
        for idx2, obj_info2 in enumerate(obj_list_query):
            if obj_info1['color']==obj_info2['color'] and obj_info1['shape']==obj_info2['shape'] and obj_info1['material']==obj_info2['material']:
                ref2query[idx1]=idx2
    if len(ref2query)!=len(obj_list_ref):
        pass
        #print('Fail to find some correspondence.')
    #assert len(ref2query)==len(obj_list_ref), "every reference object should find their corresponding objects"
    return ref2query

def load_reference_ftr(ref_dir, ref_track_dir, sim_str, ann_query, args):
    if not args.proposal_flag:
        sub_dir_list = os.listdir(ref_dir)
        sub_dir_list = sorted(sub_dir_list)
        if len(sub_dir_list) >4:
            sub_dir_list = sub_dir_list[:4]
    else:
        sub_dir_list = sorted(glob.glob(ref_dir+'*.json'))
    obj_ftr_list = []
    edge_list = []
    ref2query_list = []
    for idx, sub_dir in enumerate(sub_dir_list):
        if not args.proposal_flag:
            full_ann_dir = os.path.join(ref_dir, sub_dir) 
            ann_path = os.path.join(ref_dir, sub_dir, 'annotations', 'annotation.json')
        else:
            ann_path = sub_dir
            sub_dir = ann_path.split('_')[-1].split('.')[0]

        if not os.path.isfile(ann_path):
            print("Warning! Fail to find %s\n"%(ann_path))
            continue 
        with open(ann_path, 'r') as fh:
            ann = json.load(fh)
        ref2query = map_ref_to_query(ann_query['config'], ann['config']) 
        #if len(ref2query)!=len(ann['config']):
        #    print(sim_str)
        #    print(sub_dir)
        #    print(ref_dir)
        #    print(ref_track_dir)
            #pdb.set_trace()
        visible_list = list(ref2query.values())
        shape_emb = [ get_one_hot_for_shape(obj_info['shape']) for obj_info in ann['config']]
        shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
        shape_mat_exp = np.repeat(shape_mat, args.num_vis_frm, axis=1)
        track_path = os.path.join(ref_track_dir, sim_str+ '_' + sub_dir  +'.npy')
        track, vel = load_obj_track(track_path, args.num_vis_frm)

        obj_ftr = np.concatenate([shape_mat_exp, track, vel], axis=2)
        # align the reference objects with the target object
        obj_num_ori = len(ann_query['config'])
        obj_ftr_pad  = -1 * np.ones((obj_num_ori, obj_ftr.shape[1], obj_ftr.shape[2]), dtype=np.float32)
        for idx1, idx2 in ref2query.items():
            obj_ftr_pad[idx2] = obj_ftr[idx1]
        # Only shows the labels for the visible objects
        edge = get_edge_rel(ann_query['config'], visible_list)
        obj_ftr_pad = obj_ftr_pad.astype(np.float32)
        edge = edge.astype(np.long)
        obj_ftr_pad = torch.from_numpy(obj_ftr_pad)
        edge = torch.from_numpy(edge)
        obj_ftr_list.append(obj_ftr_pad)
        edge_list.append(edge)
        ref2query_list.append(ref2query)
    return obj_ftr_list, edge_list, ref2query_list

def load_reference_ftr_sim(ref_dir, sim_str, ann_query, args):
    fn_name = os.path.join(ref_dir, sim_str+'_*.json')
    fn_list = glob.glob(fn_name)
    fn_list = sorted(fn_list)
    obj_ftr_list = []
    edge_list = []
    ref2query_list = []
    for idx, fn in enumerate(fn_list):
        with open(fn, 'r') as fh:
            ann = json.load(fh)
        ref2query = map_ref_to_query(ann_query['config'], ann['config']) 
        visible_list = list(ref2query.values())
        shape_emb = [ get_one_hot_for_shape(obj_info['shape']) for obj_info in ann['config']]
        shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
        shape_mat_exp = np.repeat(shape_mat, args.num_vis_frm, axis=1)
        track, vel = sample_obj_track(ann['motion'], args.num_vis_frm, args.sample_every)
        # field info
        field = get_field_info_sim(track, ann_query, args.add_field_flag)
        # num_obj * num_vis_frm * 2+2+3
        obj_ftr = np.concatenate([shape_mat_exp, track, vel, field], axis=2)
        # align the reference objects with the target object
        obj_num_ori = len(ann_query['config'])
        obj_ftr_pad  = -1 * np.ones((obj_num_ori, obj_ftr.shape[1], obj_ftr.shape[2]), dtype=np.float32)
        for idx1, idx2 in ref2query.items():
            obj_ftr_pad[idx2] = obj_ftr[idx1]
        # Only shows the labels for the visible objects
        edge = get_edge_rel(ann_query['config'], visible_list)
        obj_ftr_pad = obj_ftr_pad.astype(np.float32)
        edge = edge.astype(np.long)
        obj_ftr_pad = torch.from_numpy(obj_ftr_pad)
        edge = torch.from_numpy(edge)
        obj_ftr_list.append(obj_ftr_pad)
        edge_list.append(edge)
        ref2query_list.append(ref2query)
    return obj_ftr_list, edge_list, ref2query_list

def collect_fun(data_list):
    return data_list

def build_dataloader(args, phase='train', sim_st_idx=0, sim_ed_idx=100):
    shuffle_flag = True if phase=='train' else False
    #print('DEBUG!')
    #pdb.set_trace()
    data_aug_flag = True if phase=='train' and args.data_noise_aug  else False
    ref_aug_flag = True if phase=='train' and args.ref_num_aug  else False
    mask_aug_prob = args.mask_aug_prob if phase=='train' and args.mask_aug_prob  else 0
    dataset = clevrerDataset(args, sim_st_idx, sim_ed_idx, phase,  data_aug_flag, ref_aug_flag, mask_aug_prob)
    data_loader = DataLoader(dataset,  num_workers=args.num_workers, batch_size=args.batch_size,
            shuffle=shuffle_flag, collate_fn=collect_fun)
    return data_loader

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers for the dataset.')
    parser.add_argument('--ann_dir', type=str, default="../../../render/output/causal_sim_v9_3_1",
                    help='directory for target video annotation')
    parser.add_argument('--ref_dir', type=str, default="../../../render/output/reference_v9_3_1",
                    help='directory for reference video annotation.')
    parser.add_argument('--ref_num', type=int, default=4,
            help='number of reference videos for a target video')
    parser.add_argument('--batch_size', type=int, default=2,  help='')
    parser.add_argument('--track_dir', type=str, default="../../../render/output/box_causal_sim_v9_3_1",
                    help='directory for target track annotation')
    parser.add_argument('--ref_track_dir', type=str, default="../../../render/output/box_reference_v9",
                    help='directory for reference track annotation')
    parser.add_argument('--num_vis_frm', type=int, default=125,
                    help='Number of visible frames.')
    parser.add_argument('--load_reference_flag', type=int, default=0,
                    help='Load reference videos for prediction.')
    parser.add_argument('--sim_data_flag', type=int, default=1,
                    help='Flag to use simulation data.')
    args = parser.parse_args()
    train_loader = build_dataloader(args)
