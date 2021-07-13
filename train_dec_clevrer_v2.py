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
from clevrer.clevrer_dataset_v2 import build_dataloader_v2
import clevrer.utils as clevrer_utils
import pdb

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
parser.add_argument('--lr-decay', type=int, default=8,
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
parser.add_argument('--add_field_flag', type=int, default=1,
                help='flag to indicate fields')
parser.add_argument('--frame_offset', type=int, default=5,
                help='frames to predict')
parser.add_argument('--n_his', type=int, default=2,
                help='Number of hidden layers')
parser.add_argument('--n_roll', type=int, default=4,
                help='Number of rollout steps during training')
parser.add_argument('--save_str', type=str, default='',
                    help='id folder to save the model and log')
parser.add_argument('--exclude_field_video', type=int, default=0,
                help='exclude videos with fields during training')
parser.add_argument('--ann_dir_val', type=str, default="../../render/output/causal_sim_v9_3_1",
                help='directory for target video annotation')
parser.add_argument('--ref_dir_val', type=str, default="../../render/output/reference_v9_3_1",
                help='directory for reference video annotation.')
parser.add_argument('--track_dir_val', type=str, default="../../render/output/box_causal_sim_v9_3_1",
                help='directory for target track annotation')
parser.add_argument('--ref_track_dir_val', type=str, default="../../render/output/box_reference_v9",
                help='directory for reference track annotation')
parser.add_argument('--use_ref_flag', type=int, default=0,
                help='Use reference_frames to learn dynamics')
parser.add_argument('--train_st_idx2', type=int, default=0,
                help='Start index of the training videos.')
parser.add_argument('--train_ed_idx2', type=int, default=100,
                help='End index of the training videos.')
parser.add_argument('--data_noise_weight', type=float, default=0.001,
                help='add random noise for data augumentation.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    now = datetime.datetime.now()
    if len(args.save_str)==0:
        save_str = now.isoformat()
    else:
        save_str = args.save_str
    save_folder = '{}/exp_{}/'.format(args.save_folder, save_str)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'decoder.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader = build_dataloader_v2(args, phase='train', sim_st_idx=args.train_st_idx, sim_ed_idx= args.train_ed_idx)
valid_loader = build_dataloader_v2(args, phase='val', sim_st_idx=args.val_st_idx, sim_ed_idx=args.val_ed_idx)

if args.decoder == 'mlp':
    model = MLPDecoder(n_in_node=args.dims,
                        hist_win = args.n_his+1,
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

if args.load_folder:
    load_file = os.path.join(args.load_folder, 'model.pt')
    model.load_state_dict(torch.load(load_file))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

if args.cuda:
    model.cuda()


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = 0
    loss_val = 0
    mse_baseline_train = 0
    mse_baseline_val = 0
    mse_train = 0
    mse_val = 0
    count_train = 0
    count_val = 0

    model.train()
    scheduler.step()
    for batch_idx, data_list in enumerate(train_loader):
        if batch_idx % 1000 == 0:
            print('train [%d/%d]' % (batch_idx, len(train_loader)))
        output_list = []
        target_list = []
        loss =  0
        for smp_id, smp in enumerate(data_list):
            # inputs: []
            # target: []
            inputs, relations, target, sim_str = smp 
            num_atoms = inputs.shape[1]
            #pdb.set_trace()
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

            if args.cuda:
                inputs = inputs.cuda()
                rel_type_onehot = rel_type_onehot.cuda()
                rel_rec = rel_rec.cuda()
                rel_send = rel_send.cuda()
                target = target.cuda()
            else:
                inputs = inputs.contiguous()

            if args.decoder == 'rnn':
                output = model(inputs, rel_type_onehot, rel_rec, rel_send, 1,
                               burn_in=True,
                               burn_in_steps=args.timesteps - args.prediction_steps)
            else:
                '''
                print('inputs.shape', inputs.shape)
                print('rel_type_onehot.shape', rel_type_onehot.shape)
                print('rel_rec.shape', rel_rec.shape)
                print('rel_send.shape', rel_send.shape)
                print('args.n_roll', args.n_roll)
                '''

                # output: B=1 x n_obj x T=pred_steps x state_dim
                output = model(inputs, rel_type_onehot, rel_rec, rel_send, args.n_roll)

            # print('output.shape', output.shape)
            # print('target.shape', target.shape)

            tmp_loss = nll_gaussian_v2(output, target, args.var)
            loss +=tmp_loss
            mse = F.mse_loss(output, target)
            mse_baseline = F.mse_loss(
                    inputs[:, :, :, -args.dims:].repeat(1, 1, args.n_roll, 1),
                    target)
            loss_train +=tmp_loss.data.item()
            mse_train +=mse.data.item()
            mse_baseline_train +=mse_baseline.item()
            count_train +=1

        loss = loss*1.0/len(data_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print('Epoch: {:04d}'.format(epoch),
                  'Iter: {:04d}'.format(batch_idx),
                  'nll_train: {:.10f}'.format(loss),
                  'mse_train: {:.12f}'.format(mse_train/count_train),
                  'mse_baseline_train: {:.10f}'.format(mse_baseline_train/count_train),
                  'time: {:.4f}s'.format(time.time() - t))


    model.eval()
    for batch_idx, data_list in enumerate(valid_loader):
        if batch_idx % 1000 == 0:
            print('valid [%d/%d]' % (batch_idx, len(valid_loader)))
        with torch.no_grad():
            for smp_id, smp in enumerate(data_list):
                inputs, relations, target, sim_str = smp 
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

                if args.cuda:
                    inputs = inputs.cuda()
                    rel_type_onehot = rel_type_onehot.cuda()
                    rel_rec = rel_rec.cuda()
                    rel_send = rel_send.cuda()
                    target = target.cuda()
                else:
                    inputs = inputs.contiguous()
                inputs, rel_type_onehot = Variable(inputs, volatile=True), Variable(
                    rel_type_onehot, volatile=True)

                

                output = model(inputs, rel_type_onehot, rel_rec, rel_send, args.n_roll)
                loss = nll_gaussian_v2(output, target, args.var)

                mse = F.mse_loss(output, target)
                mse_baseline = F.mse_loss(
                        inputs[:, :, :, -args.dims:].repeat(1, 1, args.n_roll, 1),
                        target)

                loss_val +=loss.data.item()
                mse_val +=mse.data.item()
                mse_baseline_val +=mse_baseline.item()
                count_val +=1

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(loss_train/count_train),
          'mse_train: {:.12f}'.format(mse_train/count_train),
          'mse_baseline_train: {:.10f}'.format(mse_baseline_train/count_train),
          'nll_val: {:.10f}'.format(loss_val/count_val),
          'mse_val: {:.12f}'.format(mse_val/count_val),
          'mse_baseline_val: {:.10f}'.format(mse_baseline_val/count_val),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and loss_val/count_val < best_val_loss:
        if int(torch.__version__.split('.')[1])>=6:
            torch.save(model.state_dict(), model_file, _use_new_zipfile_serialization=False )
        else:
            torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(loss_train/count_train),
              'mse_train: {:.12f}'.format(mse_train/count_train),
              'mse_baseline_train: {:.10f}'.format(mse_baseline_train/count_train),
              'nll_val: {:.10f}'.format(loss_val/count_val),
              'mse_val: {:.12f}'.format(mse_val/count_val),
              'mse_baseline_val: {:.10f}'.format(mse_baseline_val/count_val),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return loss_val /count_val

# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
if log is not None:
    print(save_folder)
    log.close()
