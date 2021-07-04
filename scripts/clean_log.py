import os
import pdb

log_dir='logs'
sub_dir_list = os.listdir(log_dir)
sub_dir_list = sorted(sub_dir_list)
for sub_dir in sub_dir_list:
    full_dir = os.path.join(log_dir,  sub_dir)
    fn_list = os.listdir(full_dir)
    valid_flag = False
    for fn in fn_list:
        if '.pt' in fn:
            valid_flag = True
            break
    if not valid_flag:
        cmd_str = 'rm %s -r'%full_dir
        os.system(cmd_str)
pdb.set_trace()
