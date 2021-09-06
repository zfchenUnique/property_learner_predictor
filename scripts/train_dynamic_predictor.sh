GPU_ID=$1
ANN_DIR=/home/zfchen/code/output/prp_output/v16_prp_refine/config
REF_DIR=/home/zfchen/code/output/prp_output/v16_prp_reference/config
TRACK_DIR=/home/zfchen/code/output/prp_output/v16_prp_refine/tracks
REF_TRACK_DIR=/home/zfchen/code/output/prp_output/v16_prp_reference/tracks

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_dec_clevrer_v2.py \
    --load_reference_flag 1 \
    --sim_data_flag 0 \
    --train_st_idx 0 \
    --train_ed_idx 4000 \
    --train_st_idx2 6000 \
    --train_ed_idx2 10000 \
    --val_st_idx 4000 \
    --val_ed_idx 6000 \
    --dims 8 \
    --exclude_field_video 1 \
    --ann_dir ${ANN_DIR} \
    --ref_dir ${REF_DIR} \
    --track_dir ${TRACK_DIR} \
    --ref_track_dir ${REF_TRACK_DIR} \
    --ann_dir_val ${ANN_DIR} \
    --ref_dir_val ${REF_DIR} \
    --track_dir_val ${TRACK_DIR} \
    --ref_track_dir_val ${REF_TRACK_DIR} \
    --save_str n_rollout_4_multi_msg_passing_B4_LRD8_v16_no_ref_prp \
    --use_ref_flag 0 \
    --batch_size 4 \
    --num_workers 2 \
    --data_noise_weight 0 \
    --proposal_flag  1 \
    #--ann_dir ../../../output/render_output/ann_v13 \
    #--ref_dir ../../../output/render_output/reference_v13 \
