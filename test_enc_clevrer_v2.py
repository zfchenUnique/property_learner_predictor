from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules_clevrer import *
import pdb
from clevrer.clevrer_dataset import build_dataloader
import clevrer.utils as clevrer_utils
import json

#CLASS_WEIGHT=torch.FloatTensor([0.0176, 1, 0.75])
CLASS_WEIGHT=torch.FloatTensor([0.0193, 1, 0.8893])

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation, To be changed during runing.')
parser.add_argument('--num-classes', type=int, default=3,
                    help='Number of edge types.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging.')
parser.add_argument('--edge-types', type=int, default=3,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=8,
                    help='The number of dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=125,
                    help='The number of time steps per sample.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where the trained models are.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor')
parser.add_argument('--motion', action='store_true', default=False,
                    help='Use motion capture data loader.')
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
parser.add_argument('--sim_data_flag', type=int, default=0,
                help='Flag to use simulation data.')
parser.add_argument('--sample_every', type=int, default=10,
                help='Sampling rate on simulation data.')
parser.add_argument('--mass_best_flag', type=int, default=0,
                help='Use mass best model')
parser.add_argument('--charge_best_flag', type=int, default=0,
                help='Use charge best model')
parser.add_argument('--mass_num', type=int, default=2,
                help='number of mass category.')
parser.add_argument('--max_pool_mass', type=int, default=1,
                help='max pool for mass')
parser.add_argument('--add_field_flag', type=int, default=1,
                help='flag to indicate fields')
parser.add_argument('--max_pool_charge_training', type=int, default=1,
                help='max pool for charge training')
parser.add_argument('--proposal_flag', type=int, default=0,
                help='results for mask proposals and attributes')
parser.add_argument('--data_noise_aug', type=int, default=0,
                help='add random noise for data augumentation.')
parser.add_argument('--data_noise_weight', type=float, default=0.001,
                help='add random noise for data augumentation.')
parser.add_argument('--ref_num_aug', type=int, default=0,
                help='add random numbers of reference videos for data augumentation.')
parser.add_argument('--save_str', type=str, default='',
                    help='id folder to save the model and log')
parser.add_argument('--version', type=str, default='',
                    help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    CLASS_WEIGHT = CLASS_WEIGHT.cuda()

def set_debugger():
    from IPython.core import ultratb
    import sys
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)
set_debugger()

log = None
# config model and meta-data. Always saves in a new folder.
save_folder = args.save_folder
meta_file = os.path.join(save_folder, 'test_metadata.pkl')
model_file = os.path.join(save_folder, 'encoder.pt')
model_file_mass = os.path.join(save_folder, 'encoder_mass.pt')
model_file_charge = os.path.join(save_folder, 'encoder_charge.pt')
save_result_path = os.path.join(save_folder, 'raw_prediction_%s.json'%(args.version))
save_result_path_mass = os.path.join(save_folder, 'raw_prediction_mass_%s.json'%(args.version))
save_result_path_charge = os.path.join(save_folder, 'raw_prediction_charge_%s.json'%(args.version))

if args.mass_best_flag:
    model_file = model_file_mass
    save_result_path = save_result_path_mass
    print('Using %s.\n'%model_file_mass)
if args.charge_best_flag:
    model_file = model_file_charge
    save_result_path = save_result_path_charge
    print('Using %s.\n'%model_file_charge)

log_file = os.path.join(save_folder, 'test_log.txt')
log = open(log_file, 'w')
pickle.dump({'args': args}, open(meta_file, "wb"))

test_loader = build_dataloader(args, phase='test', sim_st_idx=args.test_st_idx, sim_ed_idx=args.test_ed_idx)

if args.encoder == 'mlp':
    model = MLPEncoder(args.num_vis_frm * args.dims, args.hidden,
                       args.edge_types, args.mass_num, 
                       args.dropout, args.factor)
elif args.encoder == 'cnn':
    model = CNNEncoder(args.dims, args.hidden, args.edge_types,
                       args.dropout, args.factor)

def test():
    monitor = clevrer_utils.monitor_initialization(args, 'charge')
    monitor = clevrer_utils.monitor_initialization(args, 'mass', monitor)
    t = time.time()
    loss_test = []
    acc_test = []
    model.eval()
    model.load_state_dict(torch.load(model_file))
    mass_pred_label_dict = {}
    charge_pred_label_dict = {}
    for batch_idx, data_list in enumerate(test_loader):
        output_list = []
        target_list = []
        mass_list = []
        mass_label_list = []
        with torch.no_grad():
            for smp_id, smp in enumerate(data_list):
                data, target, ref2query_list, sim_str, mass_label, valid_flag = smp
                num_atoms = data.shape[1]
                # Generate off-diagonal interaction graph
                off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)
                rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
                rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
                rel_rec = torch.FloatTensor(rel_rec)
                rel_send = torch.FloatTensor(rel_send)
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                    rel_rec = rel_rec.cuda()
                    rel_send = rel_send.cuda()
                    mass_label = mass_label.cuda()
                output, pred_mass = model(data, rel_rec, rel_send)

                if args.max_prediction_flag:
                    output_pool = clevrer_utils.max_pool_prediction(output, num_atoms, ref2query_list)
                    output_list.append(output_pool.view(-1, args.num_classes))
                    target_list.append(target[0])
                else:
                    output_list.append(output.view(-1, args.num_classes))
                    target_list.append(target.view(-1))
                mass_pool = clevrer_utils.pool_mass_prediction(pred_mass, num_atoms, ref2query_list, args.max_pool_mass)
                mass_list.append(mass_pool.view(-1, args.mass_num))
                mass_label_list.append(mass_label)
                
                mass_pred_to_list = mass_pool.view(-1, args.mass_num).max(1)[1].cpu().numpy().tolist()
                mass_pred_label_dict[sim_str] = mass_pred_to_list 
                charge_pred_to_list = output_pool.view(-1, args.num_classes).cpu().numpy().tolist()
                charge_pred_label_dict[sim_str] = charge_pred_to_list 
            
            output = torch.cat(output_list, dim=0)
            target = torch.cat(target_list, dim=0)
            # Flatten batch dim
            output = output.view(-1, args.num_classes)
            target = target.view(-1)
            mass = torch.cat(mass_list, dim=0)
            mass_label = torch.cat(mass_label_list, dim=0)

            loss = F.cross_entropy(output, target, weight=CLASS_WEIGHT)
            
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc = correct*1.0 / pred.size(0)
            monitor, acc_list_charge = clevrer_utils.compute_acc_by_class(output, target, args.num_classes, monitor, 'charge')
            monitor, acc_list_mass = clevrer_utils.compute_acc_by_class(mass, mass_label, args.mass_num, monitor, 'mass')

            loss_test.append(loss.item())
            acc_test.append(acc)
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('loss_test: {:.10f}'.format(np.mean(loss_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)))
    acc_tr_charge = clevrer_utils.print_monitor(monitor, args.num_classes, 'charge')
    acc_tr_mass = clevrer_utils.print_monitor(monitor, args.mass_num, 'mass')
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
        log.flush()
        with open(save_result_path, 'w') as fh:
            json.dump({'mass': mass_pred_label_dict, 'charge': charge_pred_label_dict}, fh)
    return np.mean(acc_test)

if args.cuda:
    model.cuda()
t_total = time.time()
test()
if log is not None:
    print(save_folder)
    log.close()
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
