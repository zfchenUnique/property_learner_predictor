GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_dec_clevrer_v2.py \
    --load_reference_flag 1 \
    --sim_data_flag 0 \
    --train_st_idx 0 \
    --train_ed_idx 4000 \
    --val_st_idx 4000 \
    --val_ed_idx 6000 \
    --train_st_idx2 6000 \
    --train_ed_idx2 10000 \
    --dims 8 \
    --exclude_field_video 1 \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16/render/causal_sim  \
    --ref_dir /home/zfchen/code/output/render_output_vislab3/v16/render/reference \
    --track_dir /home/zfchen/code/output/render_output_vislab3/v16/box \
    --ref_track_dir /home/zfchen/code/output/render_output_vislab3/v16/box_reference \
    --ann_dir_val /home/zfchen/code/output/render_output_vislab3/v16/render/causal_sim \
    --ref_dir_val /home/zfchen/code/output/render_output_vislab3/v16/render/reference \
    --track_dir_val /home/zfchen/code/output/render_output_vislab3/v16/box \
    --ref_track_dir_val /home/zfchen/code/output/render_output_vislab3/v16/box_reference \
    --save_str no_field_n_rollout_4_multi_msg_passing_B4_LRD8_aug1e-3_v16_ref \
    --use_ref_flag 1 \
    --batch_size 4 \
    --num_workers 2 \
    #--ann_dir ../../../output/render_output/ann_v13 \
    #--ref_dir ../../../output/render_output/reference_v13 \
    #--track_dir ../../../output/render_output/box_v13 \
    #--ref_track_dir ../../../output/render_output/box_reference_v13 \
