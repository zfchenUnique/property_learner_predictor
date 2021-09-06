GPU_ID=$1
ANN_DIR=/home/zfchen/code/output/ns-vqa_output/v16_test_prp_refine/config
TRACK_DIR=/home/zfchen/code/output/ns-vqa_output/v16_test_prp_refine/tracks
PRED_DIR=v16_test/predictions_motion_prp
MODEL_DIR=logs/exp_n_rollout_4_multi_msg_passing_B4_LRD8_v16_no_ref_prp

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer_v3_prp.py \
    --batch_size 4 \
    --num_workers 1 \
    --load_reference_flag 1 \
    --test_st_idx 10000 \
    --test_ed_idx 12000 \
    --dims 8 \
    --sim_data_flag 0 \
    --exclude_field_video 1 \
    --vis_dir /home/zfchen/code/output/visualization/v16_test_prp_dynamics_vis3 \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16_test/prediction_prp_mass_charge \
    --track_dir ${TRACK_DIR} \
    --prediction_output_dir ${PRED_DIR} \
    --save-folder ${MODEL_DIR} \
    --visualize_flag 0 \
