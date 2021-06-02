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
from modules import *
from clevrer.clevrer_dataset import build_dataloader
import clevrer.utils as clevrer_utils
import pdb

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

test_loader = build_dataloader(args, phase='test', sim_st_idx=args.test_st_idx, sim_ed_idx=args.test_ed_idx)

if args.decoder == 'mlp':
    model = MLPDecoder(n_in_node=args.dims,
                       edge_types=args.edge_types,
                       msg_hid=args.hidden,
                       msg_out=args.hidden,
                       n_hid=args.hidden,
                       do_prob=args.dropout,
                       skip_first=args.skip_first)
elif args.decoder == 'rnn':
    model = RNNDecoder(n_in_node=args.dims,
                       edge_types=args.edge_types,
                       n_hid=args.hidden,
                       do_prob=args.dropout,
                       skip_first=args.skip_first)
if args.cuda:
    model.cuda()

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def test():
    loss_test = []
    mse_baseline_test = []
    mse_test = []
    tot_mse = 0
    tot_mse_baseline = 0
    counter = 0
    model.eval()
    model.load_state_dict(torch.load(model_file))
    for batch_idx, data_list in enumerate(test_loader):
        with torch.no_grad():
            for smp_id, smp in enumerate(data_list):
                inputs, relations, ref2query_list, sim_str = smp[0], smp[1], smp[2], smp[3]
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

                if args.fully_connected:
                    zeros = torch.zeros(
                        [rel_type_onehot.size(0), rel_type_onehot.size(1)])
                    ones = torch.ones(
                        [rel_type_onehot.size(0), rel_type_onehot.size(1)])
                    rel_type_onehot = torch.stack([zeros, ones], -1)

                assert (inputs.size(2) - args.timesteps) >= args.timesteps

                if args.cuda:
                    inputs = inputs.cuda()
                    rel_type_onehot = rel_type_onehot.cuda()
                    rel_rec = rel_rec.cuda()
                    rel_send = rel_send.cuda()
                else:
                    inputs = inputs.contiguous()

                ins_cut = inputs[:, :, -args.timesteps:, :].contiguous()

                output = model(ins_cut, rel_type_onehot, rel_rec, rel_send, 1)

                target = ins_cut[:, :, 1:, :]

                loss = nll_gaussian(output, target, args.var)

                mse = F.mse_loss(output, target)
                mse_baseline = F.mse_loss(ins_cut[:, :, :-1, :], ins_cut[:, :, 1:, :])

                loss_test.append(loss.item())
                mse_test.append(mse.data.item())
                mse_baseline_test.append(mse_baseline.data.item())

                # For plotting purposes
                if args.decoder == 'rnn':
                    output = model(inputs, rel_type_onehot, rel_rec, rel_send, 100,
                                   burn_in=True, burn_in_steps=args.timesteps)
                    output = output[:, :, args.timesteps:, :]
                    target = inputs[:, :, args.timesteps+1:, :]
                    baseline = inputs[:, :, -(args.timesteps + 1):-args.timesteps,
                               :].expand_as(target)
                else:
                    data_plot = inputs[:, :, args.timesteps:args.timesteps + 21,
                                :].contiguous()
                    output = model(data_plot, rel_type_onehot, rel_rec, rel_send, 20)
                    target = data_plot[:, :, 1:, :]
                    baseline = inputs[:, :, args.timesteps:args.timesteps + 1,
                               :].expand_as(target)
                if args.visualize_flag:
                    if not os.path.isdir(args.vis_dir):
                        os.makedirs(args.vis_dir)
                    sim_str = os.path.join(args.vis_dir, sim_str)
                    plot_sample(target[:, :, :10], sim_str=sim_str+'_gt')
                    plot_sample(output[:, :, :10], sim_str=sim_str+'_pred')
                    pdb.set_trace()
                mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
                tot_mse += mse.data.cpu().numpy()
                counter += 1

                mse_baseline = ((target - baseline) ** 2).mean(dim=0).mean(dim=0).mean(
                    dim=-1)
                tot_mse_baseline += mse_baseline.data.cpu().numpy()

    mean_mse = tot_mse / counter
    mse_str = '['
    for mse_step in mean_mse[:-1]:
        mse_str += " {:.12f} ,".format(mse_step)
    mse_str += " {:.12f} ".format(mean_mse[-1])
    mse_str += ']'

    mean_mse_baseline = tot_mse_baseline / counter
    mse_baseline_str = '['
    for mse_step in mean_mse_baseline[:-1]:
        mse_baseline_str += " {:.12f} ,".format(mse_step)
    mse_baseline_str += " {:.12f} ".format(mean_mse_baseline[-1])
    mse_baseline_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(loss_test)),
          'mse_test: {:.12f}'.format(np.mean(mse_test)),
          'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)))
    print('MSE: {}'.format(mse_str))
    print('MSE Baseline: {}'.format(mse_baseline_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(loss_test)),
              'mse_test: {:.12f}'.format(np.mean(mse_test)),
              'mse_baseline_test: {:.10f}'.format(np.mean(mse_baseline_test)),
              file=log)
        print('MSE: {}'.format(mse_str), file=log)
        print('MSE Baseline: {}'.format(mse_baseline_str), file=log)
        log.flush()


# Train model
t_total = time.time()
log.flush()
test()
if log is not None:
    print(save_folder)
    log.close()
