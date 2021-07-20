import argparse
import json
import numpy as np
import os
import pdb
from torch.utils.data import DataLoader, Dataset
import torch
import glob

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
    track_pad[:, :, 0] += track_pad[:, :, 2]*0.5
    track_pad[:, :, 1] += track_pad[:, :, 3]*0.5

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
    charge_id_list = [ obj_id for obj_id, obj_info in enumerate(obj_list) if obj_info['charge']!=0]
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
    def __init__(self, args, sim_st_idx, sim_ed_idx, phase):
        
        if phase=='val':
            self.ann_dir = args.ann_dir_val
            self.track_dir = args.track_dir_val
            self.ref_dir = args.ref_dir_val
            self.ref_track_dir = args.ref_track_dir_val
        elif phase=='train' or phase=='test':
            self.ann_dir = args.ann_dir
            self.track_dir = args.track_dir
            self.ref_dir = args.ref_dir
            self.ref_track_dir = args.ref_track_dir

        self.ref_num = args.ref_num
        self.num_vis_frm = args.num_vis_frm
        self.sim_list = list(range(sim_st_idx, sim_ed_idx))
        if phase=='train':
            sup_list = list(range(args.train_st_idx2, args.train_ed_idx2))
            new_train_list = sorted(list(set(self.sim_list + sup_list )))
            self.sim_list = new_train_list
        #self.sim_st_idx = sim_st_idx
        #self.sim_ed_idx = sim_ed_idx
        self.args = args
        self.phase = phase
        self.valid_info_fn = '%s_valid_idx.txt'%phase
        # Always generate training and testing splits
        if not os.path.isfile(self.valid_info_fn) or 1:
            self.__generate_valid_idx_for_render_decoder()
        else:
            self.__read_valid_idx_for_render_decoder()
        if args.use_ref_flag:        
            self.valid_info_ref_fn = '%s_valid_ref_idx.txt'%phase
            self.__generate_valid_ref_idx_for_render_decoder()

    def __read_valid_idx_for_render_decoder(self):
        fin = open(self.valid_info_fn, 'r').readlines()
        self.n_valid_idx = len(fin)
        self.valid_idx = []
        self.object_tracks = {}
        self.object_anns = {}
        for idx in range(self.n_valid_idx):
            a = int(fin[idx].strip().split(' ')[0])
            b = int(fin[idx].strip().split(' ')[1])
            self.valid_idx.append((a, b))
        for idx, sim_id in enumerate(self.sim_list):
            if idx % 500 ==0:
                print('preparing the %d/%d videos\n'%(idx, len(self.sim_list)))
            sim_str = 'sim_%05d'%(sim_id)
            # load object track
            track_path = os.path.join(self.track_dir, sim_str+'.npy')
            track, vel = load_obj_track(track_path, self.num_vis_frm)
            self.object_tracks[sim_id] = track
            # object ann
            ann_path = os.path.join(self.ann_dir, sim_str, 'annotations', 'annotation.json')
            with open(ann_path, 'r') as fh:
                ann = json.load(fh)
                self.object_anns[sim_id] = {'config': ann['config']}

    def __generate_valid_ref_idx_for_render_decoder(self):
        frame_offset = self.args.frame_offset
        if not hasattr(self, 'valid_idx'):
            self.valid_idx = []
        self.n_valid_idx = len(self.valid_idx)
        self.object_ref_tracks = {}
        self.object_ref_anns = {}
        fout = open(self.valid_info_ref_fn, 'w')
        for idx, sim_id in enumerate(self.sim_list):
            if idx % 500 ==0:
                print('preparing the %d/%d videos\n'%(idx, len(self.sim_list)))
            sim_str = 'sim_%05d'%(sim_id)
            self.object_ref_anns[sim_id] = {}
            self.object_ref_tracks[sim_id] = {} 
            for ref_id in range(self.ref_num):
                # object ann
                ann_path = os.path.join(self.ref_dir, sim_str, str(ref_id), 'annotations', 'annotation.json')
                with open(ann_path, 'r') as fh:
                    ann = json.load(fh)
                    if self.args.exclude_field_video and len(ann['field_config'])  >0:
                        continue
                    self.object_ref_anns[sim_id][ref_id] = {'config': ann['config']}
                # load object track
                track_path = os.path.join(self.ref_track_dir, sim_str+'_'+str(ref_id)+'.npy')
                track, vel = load_obj_track(track_path, self.num_vis_frm)
                self.object_ref_tracks[sim_id][ref_id] = track

                num_obj, num_frm, box_dim = track.shape
                valid_flag = np.zeros((num_obj, num_frm), dtype=np.int8)
                for dim_id in range(box_dim):
                    valid_flag_tmp1 = np.array(track[:, :, dim_id]>0, dtype=np.int8)
                    valid_flag_tmp2 = np.array(track[:, :, dim_id]<1, dtype=np.int8)
                    valid_flag +=valid_flag_tmp1
                    valid_flag +=valid_flag_tmp2
                box_flag = valid_flag == (box_dim*2) 

                n_his = self.args.n_his
                n_roll = self.args.n_roll
                frame_offset = self.args.frame_offset
                for idx2 in range(
                        n_his * frame_offset,
                        self.num_vis_frm - n_roll * frame_offset):
                    valid = True if box_flag[:, idx2].sum()>0 else False
                    if not valid:
                        continue
                    obj_appear = box_flag[:, idx2]
                    # check history windows are valid
                    for idx3 in range(n_his):
                        idx_ref = idx2 - (idx3+1) * frame_offset
                        consistence = (obj_appear == box_flag[:, idx_ref]).sum()==num_obj
                        if not consistence:
                            valid = False
                            break
                    # check future windows are valid
                    for idx3 in range(n_roll):
                        idx_next = idx2 + frame_offset * (idx3+1)
                        consistence = (obj_appear == box_flag[:, idx_next]).sum()==num_obj
                        if not consistence:
                            valid = False
                            break 
                    if valid:
                        self.valid_idx.append((sim_id, idx2, ref_id))
                        fout.write("%d %d %d\n"%(sim_id, idx2, ref_id))
                        self.n_valid_idx  +=1
        fout.close()

    def __generate_valid_idx_for_render_decoder(self):
        if not hasattr(self, 'valid_idx'):
            self.valid_idx = []
        frame_offset = self.args.frame_offset
        self.n_valid_idx = len(self.valid_idx)
        self.object_tracks = {}
        self.object_anns = {}
        fout = open(self.valid_info_fn, 'w')
        for idx, sim_id in enumerate(self.sim_list):
            if idx % 500 ==0:
                print('preparing the %d/%d videos\n'%(idx, len(self.sim_list)))
            sim_str = 'sim_%05d'%(sim_id)
            # object ann
            ann_path = os.path.join(self.ann_dir, sim_str, 'annotations', 'annotation.json')
            with open(ann_path, 'r') as fh:
                ann = json.load(fh)
                if self.args.exclude_field_video and len(ann['field_config'])  >0:
                    continue
                self.object_anns[sim_id] = {'config': ann['config']}
            # load object track
            track_path = os.path.join(self.track_dir, sim_str+'.npy')
            track, vel = load_obj_track(track_path, self.num_vis_frm)
            self.object_tracks[sim_id] = track

            num_obj, num_frm, box_dim = track.shape
            valid_flag = np.zeros((num_obj, num_frm), dtype=np.int8)
            for dim_id in range(box_dim):
                valid_flag_tmp1 = np.array(track[:, :, dim_id]>0, dtype=np.int8)
                valid_flag_tmp2 = np.array(track[:, :, dim_id]<1, dtype=np.int8)
                valid_flag +=valid_flag_tmp1
                valid_flag +=valid_flag_tmp2
            box_flag = valid_flag == (box_dim*2) 

            n_his = self.args.n_his
            n_roll = self.args.n_roll
            frame_offset = self.args.frame_offset
            for idx2 in range(
                    n_his * frame_offset,
                    self.num_vis_frm - n_roll * frame_offset):
                valid = True if box_flag[:, idx2].sum()>0 else False
                if not valid:
                    continue
                obj_appear = box_flag[:, idx2]
                # check history windows are valid
                for idx3 in range(n_his):
                    idx_ref = idx2 - (idx3+1) * frame_offset
                    consistence = (obj_appear == box_flag[:, idx_ref]).sum()==num_obj
                    if not consistence:
                        valid = False
                        break
                # check future windows are valid
                for idx3 in range(n_roll):
                    idx_next = idx2 + frame_offset * (idx3+1)
                    consistence = (obj_appear == box_flag[:, idx_next]).sum()==num_obj
                    if not consistence:
                        valid = False
                        break 
                if valid:
                    self.valid_idx.append((sim_id, idx2))
                    fout.write("%d %d\n"%(sim_id, idx2))
                    self.n_valid_idx  +=1
        fout.close()

    def __len__(self):
        # sample 1/10 data for each epoch
        return len(self.valid_idx) #// 10

    def __getitem__(self, index):
        if self.args.sim_data_flag:
            return self.__getitem_sim__(index)
        else:
            return self.__getitem_render__(index)
    
    def __getitem_render__(self, idx):
        """
        edge: of shape (num_obj * (num_obj-1))
        """
        n_his = self.args.n_his
        n_roll = self.args.n_roll

        state_dim = self.args.dims
        frame_offset = self.args.frame_offset
        if len(self.valid_idx[idx])==2:
            sim_id, idx_frame = self.valid_idx[idx][0], self.valid_idx[idx][1]
        elif len(self.valid_idx[idx])==3:
            sim_id, idx_frame, ref_id = self.valid_idx[idx][0], self.valid_idx[idx][1], self.valid_idx[idx][2]

        sim_str = 'sim_%05d'%(sim_id)
        if len(self.valid_idx[idx])==2:
            ann = self.object_anns[sim_id]
        else:
            ann = self.object_ref_anns[sim_id][ref_id]
        shape_emb = [ get_one_hot_for_shape(obj_info['shape']) for obj_info in ann['config']]
        shape_mat = np.expand_dims(np.array(shape_emb), axis=1)
        shape_mat_exp = np.repeat(shape_mat, n_his+n_roll+1, axis=1)
        # mass info
        mass_list = [obj_info['mass']==5 for obj_info in ann['config']]
        mass_label = np.expand_dims(np.array(mass_list).astype(np.float32), axis=1)
        mass_label_exp = np.repeat(mass_label, n_his+n_roll+1, axis=1)
        mass_label_exp = np.expand_dims(mass_label_exp, axis=2)
        # load object track
        objs = []
        if len(self.valid_idx[idx])==2:
            track = self.object_tracks[sim_id] 
        elif len(self.valid_idx[idx])==3:
            track = self.object_ref_tracks[sim_id][ref_id] 
        for idx2 in range(
                idx_frame - n_his * frame_offset, 
                idx_frame + frame_offset * n_roll + 1,
                frame_offset):
            obj_loc = track[:, idx2]
            objs.append(obj_loc)

        # obj_np shape: n_obj x (n_his + 1 + n_roll) x 4
        obj_np = np.stack(objs, axis=1) 
        # print('obj_np.shape', obj_np.shape)
        # Field info To be added
        # num_objs * (n_his+1+n_roll) * state_dim 
        obj_ftr = np.concatenate([shape_mat_exp, mass_label_exp, obj_np], axis=2)
        edge = get_edge_rel(ann['config'])
        obj_ftr = obj_ftr.astype(np.float32)
        edge = edge.astype(np.long)
        obj_ftr = torch.from_numpy(obj_ftr)
        edge = torch.from_numpy(edge)
        # batch size 1  & time slide 1
        x = obj_ftr[:, :n_his+1]
        if self.phase =='train':
            x[:, :, 4:] = x[:, :, 4:] + torch.randn(x[:, :, 4:].size()) * self.args.data_noise_weight
        x = x.view(1, obj_ftr.shape[0], 1, -1)
        label = obj_ftr[:, n_his+1:].view(1, obj_ftr.shape[0], n_roll, -1)
        edge = edge.view(1, -1)
        '''
        print('x.shape', x.shape)
        print('edge.shape', edge.shape)
        print('label.shape', label.shape)
        print('sim_str', sim_str)
        exit(0)
        '''
        # x: B=1 x n_obj x 1 x (n_his * state_dim)
        # label: B=1 x n_obj x n_roll x state_dim
        return x, edge, label, sim_str 

def map_ref_to_query(obj_list_query, obj_list_ref):
    ref2query ={}
    for idx1, obj_info1 in enumerate(obj_list_ref):
        for idx2, obj_info2 in enumerate(obj_list_query):
            if obj_info1['color']==obj_info2['color'] and obj_info1['shape']==obj_info2['shape'] and obj_info1['material']==obj_info2['material']:
                ref2query[idx1]=idx2
    assert len(ref2query)==len(obj_list_ref), "every reference object should find their corresponding objects"
    return ref2query

def collect_fun(data_list):
    return data_list

def build_dataloader_v2(args, phase='train', sim_st_idx=0, sim_ed_idx=100):
    shuffle_flag = True if phase=='train' else False
    dataset = clevrerDataset(args, sim_st_idx, sim_ed_idx, phase)
    data_loader = DataLoader(dataset,  num_workers=args.num_workers, batch_size=args.batch_size, shuffle=shuffle_flag, collate_fn=collect_fun)
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
