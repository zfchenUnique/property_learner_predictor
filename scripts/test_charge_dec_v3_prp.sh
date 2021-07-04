GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer_v3_prp.py \
    --batch_size 4 \
    --num_workers 1 \
    --load_reference_flag 1 \
    --test_st_idx 3000 \
    --test_ed_idx 4000 \
    --visualize_flag 0 \
    --dims 8 \
    --sim_data_flag 0 \
    --exclude_field_video 1 \
    --vis_dir ../../../output/dynamics \
    --ann_dir /home/zfchen/code/output/ns-vqa_output/v14_prp_pred_v5/config \
    --track_dir /home/zfchen/code/output/ns-vqa_output/v14_prp_refine/tracks \
    --prediction_output_dir /home/zfchen/code/output/render_output_disk2/prediction_v14_prp_v15_dynamics \
    --save-folder logs/exp_no_field_n_rollout_4_multi_msg_passing_B4_LRD8_aug1e-3_v15 \
    #--save-folder logs/exp_no_field_n_rollout_4_multi_msg_passing \
