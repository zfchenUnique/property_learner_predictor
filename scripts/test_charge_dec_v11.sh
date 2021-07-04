GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer_v3.py \
    --batch_size 4 \
    --num_workers 1 \
    --load_reference_flag 1 \
    --test_st_idx 0 \
    --test_ed_idx 5000 \
    --visualize_flag 1 \
    --dims 8 \
    --sim_data_flag 0 \
    --exclude_field_video 1 \
    --vis_dir ../../../output/dynamics \
    --ann_dir ../../../output/render_output/causal_sim_v11_4 \
    --ref_dir ../../../output/render_output/reference_v11_4 \
    --track_dir ../../../output/render_output/box_v11 \
    --ref_track_dir ../../../output/render_output/box_reference_v11 \
    --save-folder logs/exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8_aug1e-3_v15_ref \
    --prediction_output_dir /home/zfchen/code/output/render_output_disk2/prediction_v11_v3 \
    #--save-folder logs/exp_no_field_n_rollout_4_multi_msg_passing \
    # --save-folder logs/exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8 \
    #--save-folder logs/exp_no_field_n_rollout_4 \
    #--ann_dir ../../../output/render_output/ann_v13 \
    #--ref_dir ../../../output/render_output/reference_v13 \
    #--track_dir ../../../output/render_output/box_v13 \
    #--ref_track_dir ../../../output/render_output/box_reference_v13 \
    #--save-folder logs/exp_no_field \
    #--save-folder logs/exp2021-05-11T16:42:32.009489 \
