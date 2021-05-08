from matplotlib import cm
from numpy import linspace
import numpy as np

def plot_sample(pyt_track, loc_dim_st=3, sim_str):
    pdb.set_trace()
    np_track  = pyt_track.cpu().data.numpy()
    vid_num = np_track.shape[0]
    for vid in range(vid_num):
        save_id = sim_str +'_query' if vid==0 else sim_str + '_%d'
        plot_trajectories(np_track[vid], loc_dim_st, save_id)

def plot_trajectories(ftr_ori, loc_dim_st=3, save_id):
    """
    ftr_ori: num_timesteps x num_objs x ftr_dim 
    """
    ftr_loc = ftr_ori[:, :, loc_dim_st:loc_dim_st+2]
    loc = np.transpose(ftr_loc, [1, 2, 0])
    plot_loc(loc)

def plot_loc(loc):
    """
    loc: num_timesteps x num_dims x num_objects
    """
    start = 0.0
    stop = 1.0
    num_colors = 10
    cm_subsection = linspace(start, stop, num_colors) 

    colors = [ cm.Set1(x) for x in cm_subsection ]

    for i in range(loc.shape[-1]):
        for t in range(loc.shape[0]):
            # Plot fading tail for past locations.
            plt.plot(loc[t, 0, i], loc[t, 1, i], 'o', markersize=3, 
                     color=colors[i], alpha=1-(float(t)/loc.shape[0]))
        # Plot final location.
        plt.plot(loc[-1, 0, i], loc[-1, 1, i], 'o', color=colors[i])
