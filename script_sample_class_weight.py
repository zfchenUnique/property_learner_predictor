import argparse
import pdb
from clevrer.clevrer_dataset import build_dataloader
import torch

parser = argparse.ArgumentParser()
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
parser.add_argument('--load_reference_flag', type=int, default=1,
                help='Load reference videos for prediction.')
parser.add_argument('--max_prediction_flag', type=int, default=1,
                help='Load reference videos for prediction.')
parser.add_argument('--num-classes', type=int, default=3,
                    help='Number of edge types.')
parser.add_argument('--sim_data_flag', type=int, default=1,
                help='Flag to use simulation data.')
parser.add_argument('--sample_every', type=int, default=10,
                help='Sampling rate on simulation data.')

args = parser.parse_args()
train_loader = build_dataloader(args, phase='train', sim_st_idx=args.train_st_idx, sim_ed_idx= args.train_ed_idx)

class_freq = [0 for idx in range(args.num_classes)]
for batch_idx, data_list in enumerate(train_loader):
    for smp in data_list:
        data, target = smp[0], smp[1]
        for cls_id in range(args.num_classes):
            num = torch.sum(target==cls_id)
            class_freq[cls_id] +=num
total_num = sum(class_freq)
print('total num:%d\n'%total_num)
print('class fre:\n')
print(class_freq)
print('class weight:\n')
min_cls = min(class_freq)
class_w = [min_cls*1.0 /ele for ele in class_freq]
print(class_w)
