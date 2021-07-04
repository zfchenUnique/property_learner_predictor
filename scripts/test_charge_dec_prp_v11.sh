GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer_v3_prp.py \
    --batch_size 4 \
    --num_workers 1 \
    --load_reference_flag 1 \
    --test_st_idx 0 \
    --test_ed_idx 5000 \
    --visualize_flag 0 \
    --dims 8 \
    --sim_data_flag 0 \
    --exclude_field_video 1 \
    --vis_dir ../../../output/dynamics \
    --ann_dir /home/zfchen/code/output/ns-vqa_output/v11_prp_pred_v2_debug/config \
    --track_dir /home/zfchen/code/output/ns-vqa_output/v11_prp_refine_debug/tracks \
    --prediction_output_dir /home/zfchen/code/output/render_output_disk2/prediction_v11_prp_debug \
    --save-folder logs/exp_no_field_n_rollout_4_multi_msg_passing \
