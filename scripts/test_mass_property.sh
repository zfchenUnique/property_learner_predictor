GPU_ID=$1
ANN_DIR=/home/zfchen/code/output/prp_output/v16_prp_refine/config
REF_DIR=/home/zfchen/code/output/prp_output/v16_prp_reference/config
TRACK_DIR=/home/zfchen/code/output/prp_output/v16_prp_refine/tracks
REF_TRACK_DIR=/home/zfchen/code/output/prp_output/v16_prp_reference/tracks

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_enc_clevrer_v2.py \
    --load_reference_flag 1 \
    --test_st_idx 4000 \
    --test_ed_idx 6000 \
    --dims 11 \
    --sim_data_flag 0 \
    --num_workers 2 \
    --batch_size 2 \
    --ann_dir ${ANN_DIR} \
    --ref_dir ${REF_DIR} \
    --track_dir ${TRACK_DIR} \
    --ref_track_dir ${REF_TRACK_DIR} \
    --version v16_val_prp \
    --proposal_flag 1 \
    --mass_best_flag 1 \
    --save-folder  logs/exp_v16_mass_noise_001_prp \
