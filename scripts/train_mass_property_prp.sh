GPU_ID=$1
ANN_DIR=/home/zfchen/code/output/prp_output/v16_prp_refine/config
REF_DIR=/home/zfchen/code/output/prp_output/v16_prp_reference/config
TRACK_DIR=/home/zfchen/code/output/prp_output/v16_prp_refine/tracks
REF_TRACK_DIR=/home/zfchen/code/output/prp_output/v16_prp_reference/tracks

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_enc_clevrer.py \
    --train_st_idx 0 \
    --train_ed_idx 4000 \
    --train_st_idx2 6000 \
    --train_ed_idx2 10000 \
    --val_st_idx 4000 \
    --val_ed_idx 6000 \
    --dims 11 \
    --sim_data_flag 0 \
    --batch_size 64 \
    --ann_dir ${ANN_DIR} \
    --ref_dir ${REF_DIR} \
    --track_dir ${TRACK_DIR} \
    --ref_track_dir ${REF_TRACK_DIR} \
    --ann_dir_val ${ANN_DIR} \
    --ref_dir_val ${REF_DIR} \
    --track_dir_val ${TRACK_DIR} \
    --ref_track_dir_val ${REF_TRACK_DIR} \
    --data_noise_aug 1 \
    --data_noise_weight 0.001 \
    --num_workers 4 \
    --mask_aug_prob 0 \
    --load_reference_flag 1 \
    --ref_num_aug 1 \
    --proposal_flag  1 \
    --save_str v16_mass_noise_001_prp \
    --mass_only_flag 1 \
    --charge_only_flag 0 \
