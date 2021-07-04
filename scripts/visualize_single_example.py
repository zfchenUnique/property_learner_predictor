import json  
import os
import pdb
from utils import *
import numpy  as np

def visualize_example():
    #prediction_output_dir = "/home/zfchen/code/output/render_output_disk2/prediction_v14_4"
    prediction_output_dir = "/home/zfchen/code/output/render_output_disk2/prediction_v14_prp"
    vis_dir = "/home/zfchen/code/output/visualization"
    sim_id = 3001
    sim_str = 'sim_%05d'%(sim_id)
    pred_full_path = os.path.join(prediction_output_dir, 'sim_%05d.json'%(sim_id))
    fh = open(pred_full_path, 'r')
    fd = json.load(fh)
    mass_id = 0
    what_if =  1
    charge_id = 0
    for pred_key in ['mass', 'charge']:
        pred_list = fd[pred_key]
        for pred in pred_list:
            if pred['what_if']!=what_if:
                continue
            #if 'mass' not in pred or pred['mass']!=mass_id:
            #    continue
            if 'charge' not in pred or pred['charge']!=charge_id:
                continue
            sim_str_full = os.path.join(vis_dir, sim_str+'_'+str(what_if)+'_'+str(mass_id)+'_'+str(charge_id) +'_new' )
            objs_pred = np.array(pred['trajectories'])
            #pdb.set_trace()
            plot_video_trajectories(objs_pred, loc_dim_st=0, save_id=sim_str_full+'_prp')
    #pdb.set_trace()

if __name__ == '__main__':
    visualize_example()
