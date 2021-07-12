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
torch.autograd.set_detect_anomaly(True)


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
                    help='Where to save the trained model.')
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
parser.add_argument('--sim_data_flag', type=int, default=1,
                help='Flag to use simulation data.')
parser.add_argument('--sample_every', type=int, default=10,
                help='Sampling rate on simulation data.')
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
parser.add_argument('--save_str', type=str, default='',
                    help='id folder to save the model and log')
parser.add_argument('--data_noise_aug', type=int, default=0,
                help='add random noise for data augumentation.')
parser.add_argument('--data_noise_weight', type=float, default=0.001,
                help='add random noise for data augumentation.')
parser.add_argument('--ref_num_aug', type=int, default=0,
                help='add random numbers of reference videos for data augumentation.')
parser.add_argument('--mass_only_flag', type=int, default=0,
                help='Flag to use mass only to supervise')
parser.add_argument('--mask_aug_prob', type=float, default=0,
                help='mask out trajectories to make predictions')
parser.add_argument('--charge_only_flag', type=int, default=0,
                help='Flag to use mass only to supervise')
parser.add_argument('--ann_dir_val', type=str, default="../../render/output/causal_sim_v9_3_1",
                help='directory for target video annotation')
parser.add_argument('--ref_dir_val', type=str, default="../../render/output/reference_v9_3_1",
                help='directory for reference video annotation.')
parser.add_argument('--track_dir_val', type=str, default="../../render/output/box_causal_sim_v9_3_1",
                help='directory for target track annotation')
parser.add_argument('--ref_track_dir_val', type=str, default="../../render/output/box_reference_v9",
                help='directory for reference track annotation')
parser.add_argument('--light_weight', type=float, default=0.15,
                help='class weight for light objects')
parser.add_argument('--train_st_idx2', type=int, default=0,
                help='Start index of the training videos.')
parser.add_argument('--train_ed_idx2', type=int, default=100,
                help='End index of the training videos.')
parser.add_argument('--uncharge_weight', type=float, default=0.025,
                help='class weight for uncharged objects')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

# A pre-define class weight for class balance during calculating loss
MASS_WEIGHT=torch.FloatTensor([args.light_weight, 1.0])
CHARGE_WEIGHT=torch.FloatTensor([args.uncharge_weight, 1.0, 1.0])

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    CHARGE_WEIGHT = CHARGE_WEIGHT.cuda()
    MASS_WEIGHT = MASS_WEIGHT.cuda()

def set_debugger():
    from IPython.core import ultratb
    import sys
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)
set_debugger()

log = None
# Save model and meta-data. Always saves in a new folder.
if args.save_folder:
    if len(args.save_str)==0:
        save_str = now.isoformat()
    else:
        save_str = args.save_str
    save_folder = '{}/exp_{}/'.format(args.save_folder, save_str)
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, 'encoder.pt')
    model_file_mass = os.path.join(save_folder, 'encoder_mass.pt')
    model_file_charge = os.path.join(save_folder, 'encoder_charge.pt')
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader = build_dataloader(args, phase='train', sim_st_idx=args.train_st_idx, sim_ed_idx= args.train_ed_idx)
valid_loader = build_dataloader(args, phase='val', sim_st_idx=args.val_st_idx, sim_ed_idx=args.val_ed_idx)

if args.encoder == 'mlp':
    model = MLPEncoder(args.num_vis_frm * args.dims, args.hidden,
                       args.edge_types, args.mass_num, 
                       args.dropout, args.factor)
elif args.encoder == 'cnn':
    model = CNNEncoder(args.dims, args.hidden, args.edge_types,
                       args.dropout, args.factor)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

if args.cuda:
    model.cuda()

best_model_params = model.state_dict()

def train(epoch, best_val_accuracy, best_val_accuracy_mass, best_val_accuracy_charge):
    t = time.time()
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    loss_charge_train = []
    loss_charge_val = []
    loss_mass_train = []
    loss_mass_val = []
    model.train()
    scheduler.step()
    monitor = clevrer_utils.monitor_initialization(args, 'charge')
    monitor = clevrer_utils.monitor_initialization(args, 'mass', monitor)
    for batch_idx, data_list in enumerate(train_loader):
        # since video may be with different object numbers, feed it one by one
        output_list = []
        target_list = []
        mass_list = []
        mass_label_list = []
        optimizer.zero_grad()
        for smp in data_list:
            data, target,ref2query_list, sim_str, mass_label, valid_flag = smp
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
        
            if not args.max_pool_charge_training:
                output_list.append(output.view(-1, args.num_classes))
                target_list.append(target.view(-1))
            else:
                output_pool = clevrer_utils.max_pool_prediction(output, num_atoms, ref2query_list)
                output_list.append(output_pool.view(-1, args.num_classes))
                target_list.append(target[0])
            
            mass_pool = clevrer_utils.pool_mass_prediction(pred_mass, num_atoms, ref2query_list, args.max_pool_mass)
            mass_list.append(mass_pool.view(-1, args.mass_num))
            mass_label_list.append(mass_label.view(-1))

            monitor, acc_list = clevrer_utils.compute_acc_by_class(output_pool, target[0], args.num_classes, monitor, 'charge')
            monitor, acc_list_mass = clevrer_utils.compute_acc_by_class(mass_pool, mass_label, args.mass_num, monitor, 'mass')

        output_cat = torch.cat(output_list, dim=0)
        target_cat = torch.cat(target_list, dim=0)
        mass_cat = torch.cat(mass_list, dim=0)
        mass_label_cat = torch.cat(mass_label_list, dim=0)

        loss_charge = F.cross_entropy(output_cat, target_cat, weight=CHARGE_WEIGHT)
        loss_mass = F.cross_entropy(mass_cat, mass_label_cat, weight=MASS_WEIGHT)
        if args.mass_only_flag:
            loss  = loss_mass
        elif args.charge_only_flag:
            loss  = loss_charge
        else:
            loss  = loss_mass + loss_charge 
        loss.backward()
        optimizer.step()

        pred = output_cat.data.max(1, keepdim=True)[1]
        correct = pred.eq(target_cat.data.view_as(pred)).cpu().sum()
        acc = correct*1.0 / pred.size(0)

        loss_train.append(loss.item())
        loss_charge_train.append(loss_charge.item())
        loss_mass_train.append(loss_mass.item())
        acc_train.append(acc)
        
        #if batch_idx % 100==0:
        #    print('Training: batch id %d\n'%(batch_idx))
        #    acc_tr = clevrer_utils.print_monitor(monitor, args.num_classes)
    acc_tr_charge = clevrer_utils.print_monitor(monitor, args.num_classes, 'charge')
    acc_tr_mass = clevrer_utils.print_monitor(monitor, args.mass_num, 'mass')
    acc_tr = 0.5 * (acc_tr_charge + acc_tr_mass)
    model.eval()
    #monitor = clevrer_utils.monitor_initialization(args)
    monitor = clevrer_utils.monitor_initialization(args, 'charge')
    monitor = clevrer_utils.monitor_initialization(args, 'mass', monitor)
    for batch_idx, data_list in enumerate(valid_loader):
        output_list = []
        target_list = []
        mass_list = []
        mass_label_list = []
        with torch.no_grad():
            for smp_id, smp in enumerate(data_list):
                data, target,ref2query_list, sim_str, mass_label, valid_flag = smp
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

            output = torch.cat(output_list, dim=0)
            target = torch.cat(target_list, dim=0)
            mass = torch.cat(mass_list, dim=0)
            mass_label = torch.cat(mass_label_list, dim=0)

            loss_charge = F.cross_entropy(output, target, weight=CHARGE_WEIGHT)
            loss_mass = F.cross_entropy(mass,  mass_label, weight=MASS_WEIGHT)
            loss  =  loss_charge  + loss_mass 

            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc = correct*1.0 / pred.size(0)
            monitor, acc_list_charge = clevrer_utils.compute_acc_by_class(output, target, args.num_classes, monitor, 'charge')
            monitor, acc_list_mass = clevrer_utils.compute_acc_by_class(mass, mass_label, args.mass_num, monitor, 'mass')
            loss_val.append(loss.item())
            loss_charge_val.append(loss_charge.item())
            loss_mass_val.append(loss_mass.item())
            acc_val.append(acc)
    acc_vl_charge = clevrer_utils.print_monitor(monitor, args.num_classes, 'charge')
    acc_vl_mass = clevrer_utils.print_monitor(monitor, args.mass_num, 'mass')
    acc_vl = 0.5 * (acc_vl_charge + acc_vl_mass)
    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.10f}'.format(np.mean(loss_train)),
          'loss_train_mass: {:.10f}'.format(np.mean(loss_mass_train)),
          'loss_train_charge: {:.10f}'.format(np.mean(loss_charge_train)),
          'acc_train: {:.10f}'.format(acc_tr),
          'acc_train_mass: {:.10f}'.format(acc_tr_mass),
          'acc_train_charge: {:.10f}'.format(acc_tr_charge),
          '\n',
          'loss_val: {:.10f}'.format(np.mean(loss_val)),
          'loss_val_mass: {:.10f}'.format(np.mean(loss_mass_val)),
          'loss_val_charge: {:.10f}'.format(np.mean(loss_charge_val)),
          'acc_val: {:.10f}'.format(acc_vl),
          'acc_val_mass: {:.10f}'.format(acc_vl_mass),
          'acc_val_charge: {:.10f}'.format(acc_vl_charge),
          'time: {:.4f}s'.format(time.time() - t), file=log)
    log.flush()
    if args.save_folder and 0.5 * (acc_vl_charge + acc_vl_mass) > best_val_accuracy:
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'acc_train: {:.10f}'.format(acc_tr),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'acc_val: {:.10f}'.format( 0.5 * (acc_vl_charge+acc_vl_mass) ),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    if args.save_folder and acc_vl_mass > best_val_accuracy_mass:
        torch.save(model.state_dict(), model_file_mass)
        print('Best mass model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'acc_train: {:.10f}'.format(acc_tr),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'acc_val_mass: {:.10f}'.format(acc_vl_mass),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    if args.save_folder and acc_vl_charge > best_val_accuracy_charge:
        torch.save(model.state_dict(), model_file_charge)
        print('Best charge model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'acc_train: {:.10f}'.format(acc_tr),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'acc_val_charge: {:.10f}'.format(acc_vl_charge),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return acc_vl, acc_vl_mass, acc_vl_charge

# Train model
t_total = time.time()
best_val_accuracy = -1.
best_val_accuracy_mass = -1.
best_val_accuracy_charge = -1.
best_epoch = 0
for epoch in range(args.epochs):
    val_acc, val_acc_mass, val_acc_charge = train(epoch, best_val_accuracy, 
            best_val_accuracy_mass, best_val_accuracy_charge)
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_epoch = epoch
    if val_acc_mass > best_val_accuracy_mass:
        best_val_accuracy_mass = val_acc_mass
    if val_acc_charge > best_val_accuracy_charge:
        best_val_accuracy_charge = val_acc_charge

print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
if log is not None:
    print(save_folder)
    log.close()
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
