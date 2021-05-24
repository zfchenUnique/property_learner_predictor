from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules_clevrer import *
from clevrer.clevrer_dataset_v2 import load_obj_track, get_one_hot_for_shape, get_edge_rel
import clevrer.utils as clevrer_utils
import pdb
import json
import copy

def set_debugger():
    from IPython.core import ultratb
    import sys
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)
set_debugger()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--num-classes', type=int, default=2,
                    help='Number of edge types.')
parser.add_argument('--suffix', type=str, default='_springs',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp or rnn).')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging.')
parser.add_argument('--prediction-steps', type=int, default=1, metavar='N',
                    help='Num steps to predict before using teacher forcing.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model.')
parser.add_argument('--edge-types', type=int, default=3,
                    help='The number of edge types to infer.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor')
parser.add_argument('--motion', action='store_true', default=False,
                    help='Use motion capture data loader.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of dimensions (position + velocity).')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--fully-connected', action='store_true', default=False,
                    help='Use fully-connected graph.')
# for clevrer dataset
parser.add_argument('--num_workers', type=int, default=0,
                help='Number of workers for the dataset.')
parser.add_argument('--ann_dir', type=str, default="../../render/output/causal_sim_v9_3_1",
                help='directory for target video annotation')
parser.add_argument('--ref_dir', type=str, default="../../render/output/reference_v9_3_1",
                help='directory for reference video annotation.')
parser.add_argument('--ref_num', type=int, default=4,
        help='number of reference videos for a target video')
parser.add_argument('--batch_size', type=int, default=1,  help='')
parser.add_argument('--track_dir', type=str, default="../../render/output/box_causal_sim_v9_3_1",
                help='directory for target track annotation')
parser.add_argument('--ref_track_dir', type=str, default="../../render/output/box_reference_v9",
                help='directory for reference track annotation')
parser.add_argument('--num_vis_frm', type=int, default=125,
                help='Number of visible frames.')
parser.add_argument('--train_st_idx', type=int, default=0,
                help='Start index of the training videos.')
parser.add_argument('--train_ed_idx', type=int, default=100,
                help='End index of the training videos.')
parser.add_argument('--val_st_idx', type=int, default=100,
                help='Start index of the training videos.')
parser.add_argument('--val_ed_idx', type=int, default=120,
                help='End index of the training videos.')
parser.add_argument('--test_st_idx', type=int, default=100,
                help='Start index of the test videos.')
parser.add_argument('--test_ed_idx', type=int, default=120,
                help='End index of the test videos.')
parser.add_argument('--load_reference_flag', type=int, default=0,
                help='Load reference videos for prediction.')
parser.add_argument('--max_prediction_flag', type=int, default=1,
                help='Load reference videos for prediction.')
parser.add_argument('--sim_data_flag', type=int, default=1,
                help='Flag to use simulation data.')
parser.add_argument('--sample_every', type=int, default=10,
                help='Sampling rate on simulation data.')
parser.add_argument('--vis_dir', type=str, default="visualization",
                help='directory for visualization')
parser.add_argument('--visualize_flag', type=int, default=0,
                help='visualization flag for data track')
parser.add_argument('--add_field_flag', type=int, default=1,
                help='flag to indicate fields')
parser.add_argument('--frame_offset', type=int, default=5,
                help='frames to predict')
parser.add_argument('--n_his', type=int, default=2,
                help='Number of hidden layers')
parser.add_argument('--pred_frm_num', type=int, default=25,
                help='Number of frames to predict')
parser.add_argument('--exclude_field_video', type=int, default=0,
                help='exclude videos with fields during training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

save_folder = args.save_folder
meta_file = os.path.join(save_folder, 'test_metadata.pkl')
model_file = os.path.join(save_folder, 'decoder.pt')
log_file = os.path.join(save_folder, 'test_log.txt')
log = open(log_file, 'w')
pickle.dump({'args': args}, open(meta_file, "wb"))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

assert args.decoder =='mlp', "Current version only supports MLP decoder"

if args.decoder == 'mlp':
    model = MLPDecoder(n_in_node=args.dims,
                       hist_win=args.n_his+1,
                       edge_types=args.edge_types,
                       msg_hid=args.hidden,
                       msg_out=args.hidden,
                       n_hid=args.hidden,
                       do_prob=args.dropout,
                       skip_first=args.skip_first)
if args.cuda:
    model.cuda()

def test():
    loss_test = []
    mse_baseline_test = []
    mse_test = []
    tot_mse = 0
    tot_mse_baseline = 0
    counter = 0
    model.eval()
    print("Loading ckp from %s ..." % model_file)
    model.load_state_dict(torch.load(model_file))
    
    test_list = np.arange(args.test_st_idx, args.test_ed_idx).tolist()
    frame_offset = args.frame_offset
    n_his = args.n_his

    for test_idx in range(len(test_list)):
        sim_id = test_list[test_idx]
        sim_str = 'sim_%05d'%(sim_id)
        
        ann_path = os.path.join(args.ann_dir, sim_str, 'annotations', 'annotation.json')
        with open(ann_path, 'r') as fh:
            ann = json.load(fh)
        if args.exclude_field_video and len(ann['field_config'])  >0:
            continue
        track_path = os.path.join(args.track_dir, sim_str+'.npy')
        track, vel = load_obj_track(track_path, args.num_vis_frm)
        track = torch.from_numpy(track.astype(np.float32))
        
        shape_emb = [ get_one_hot_for_shape(obj_info['shape']) for obj_info in ann['config']]
        shape_mat = np.expand_dims(np.array(shape_emb).astype(np.float32), axis=1)
        shape_mat_exp_np = np.repeat(shape_mat, n_his+1, axis=1)
        # mass info
        mass_list = [obj_info['mass']==5 for obj_info in ann['config']]
        mass_label = np.expand_dims(np.array(mass_list).astype(np.float32), axis=1)
        mass_label_exp_np = np.repeat(mass_label, n_his+1, axis=1)
        mass_label_exp_np = np.expand_dims(mass_label_exp_np, axis=2)
        objs_gt = []
        for frm_id in range(0, track.shape[1], frame_offset):
            obj_loc = track[:, frm_id]
            objs_gt.append(obj_loc)
        objs_pred = []
        pred_st = 0
        edge = get_edge_rel(ann['config'])
        #if np.sum(edge)>0:
        #    print(sim_str)
        #   pdb.set_trace()
        #    continue
        #else:
        #    continue

        for pred_id in range(pred_st, pred_st+args.pred_frm_num):
            if len(objs_pred)<n_his + 1:
                objs_pred.append(objs_gt[pred_id])
                continue
            obj_pos_list = objs_pred[pred_id-n_his-1: pred_id]
            # num_obj x n_his+1 x box_dim
            obj_pos = torch.stack(obj_pos_list, dim=1)    
            obj_states = check_obj_inputs_valid_state(obj_pos)
            valid_obj_ids = [idx_obj for idx_obj in range(obj_states.shape[0]) if obj_states[idx_obj]]
            if len(valid_obj_ids)<=1:
                objs_pred.append(objs_gt[pred_id])
                continue
            # using only valid objects
            # print('num_obj', obj_pos.shape[0])
            # print(pred_id, valid_obj_ids)

            edge = get_edge_rel([ann['config'][obj_id] for obj_id in valid_obj_ids])
            edge = edge.astype(np.long)
            edge = torch.from_numpy(edge)
            edge = edge.view(1, -1)
            shape_mat_exp = torch.from_numpy(shape_mat_exp_np)
            mass_label_exp = torch.from_numpy(mass_label_exp_np)
            obj_pos_valid = obj_pos[valid_obj_ids]
            shape_mat_exp_valid = shape_mat_exp[valid_obj_ids]
            mass_label_exp_valid = mass_label_exp[valid_obj_ids]
            step_output = forward_step( obj_pos_valid
                    , shape_mat_exp_valid, mass_label_exp_valid, edge, model)
            frame_output = copy.deepcopy(objs_gt[pred_id])
            frame_output[valid_obj_ids] = step_output[0, :, 0].cpu()
            objs_pred.append(frame_output)

        for idx_check in range(len(objs_gt)):
            print(idx_check)
            print(objs_gt[idx_check])
            print(objs_pred[idx_check])

        # num_obj,  num_frame, box_dim
        objs_gt = torch.stack(objs_gt, dim=1) 
        objs_pred = torch.stack(objs_pred, dim=1) 
        sim_str = os.path.join(args.vis_dir, sim_str)
        plot_video_trajectories(objs_gt[:, pred_st:pred_st+args.pred_frm_num], loc_dim_st=0, save_id=sim_str+'_gt')
        plot_video_trajectories(objs_pred, loc_dim_st=0, save_id=sim_str+'_query')
        pdb.set_trace()

def check_track_state(track): 
    num_obj, num_frm, box_dim = track.shape
    valid_flag = np.zeros((num_obj, num_frm), dtype=np.int8)
    for dim_id in range(box_dim):
        valid_flag_tmp1 = np.array(track[:, :, dim_id]>0, dtype=np.int8)
        valid_flag_tmp2 = np.array(track[:, :, dim_id]<1, dtype=np.int8)
        valid_flag +=valid_flag_tmp1
        valid_flag +=valid_flag_tmp2
    box_flag = valid_flag == (box_dim*2) 
    return box_flag

def check_obj_inputs_valid_state(obj_pos):
    num_obj, frm_num, box_dim = obj_pos.shape
    box_flag = check_track_state(obj_pos)
    # make all steps are valid
    obj_valid = box_flag.sum(axis=1)==frm_num 
    return obj_valid

def forward_step(obj_pos, shape_mat_exp, mass_label_exp, relations, model):
    # print('shape_mat_exp.shape', shape_mat_exp.shape)
    # print('mass_label_exp.shape', mass_label_exp.shape)
    # print('obj_pos.shape', obj_pos.shape)
    inputs = torch.cat([shape_mat_exp, mass_label_exp, obj_pos], dim=2)
    inputs = inputs.view(1, inputs.shape[0], 1, -1)
    with torch.no_grad():
        num_atoms = inputs.shape[1]
        # Generate fully-connected interaction graph (sparse graphs would also work)
        off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)
        rel_type_onehot = torch.FloatTensor(inputs.size(0), rel_rec.size(0),
                                            args.edge_types)
        rel_type_onehot.zero_()
        rel_type_onehot.scatter_(2, relations.view(inputs.size(0), -1, 1), 1)

        '''
        print(rel_type_onehot)
        print(rel_rec)
        print(rel_send)
        '''

        if args.fully_connected:
            zeros = torch.zeros(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            ones = torch.ones(
                [rel_type_onehot.size(0), rel_type_onehot.size(1)])
            rel_type_onehot = torch.stack([zeros, ones], -1)

        if args.cuda:
            rel_type_onehot = rel_type_onehot.cuda()
            rel_rec = rel_rec.cuda()
            rel_send = rel_send.cuda()
            inputs = inputs.cuda()
            relations = relations.cuda()
        else:
            inputs = inputs.contiguous()

        '''
        print('inputs.shape', inputs.shape)
        print('rel_type_onehot.shape', rel_type_onehot.shape)
        print('rel_rec.shape', rel_rec.shape)
        print('rel_send.shape', rel_send.shape)
        '''

        output = model(inputs, rel_type_onehot, rel_rec, rel_send, 1)
        return output[:, :, :, 4:]



test()


