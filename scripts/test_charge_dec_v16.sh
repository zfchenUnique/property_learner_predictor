GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer_v3.py \
    --batch_size 4 \
    --num_workers 1 \
    --load_reference_flag 1 \
    --test_st_idx 4000 \
    --test_ed_idx 6000 \
    --visualize_flag 0 \
    --dims 8 \
    --sim_data_flag 0 \
    --exclude_field_video 1 \
    --vis_dir /home/zfchen/code/output/visualization/v16_dynamics \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16/render/causal_sim \
    --ref_dir /home/zfchen/code/output/render_output_vislab3/v16/render/reference \
    --track_dir /home/zfchen/code/output/render_output_vislab3/v16/box \
    --ref_track_dir /home/zfchen/code/output/render_output_vislab3/v16/box_reference \
    --save-folder logs/exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8_aug1e-3_v16_ref \
    --prediction_output_dir /home/zfchen/code/output/render_output_vislab3/v16/predictions_gt \
