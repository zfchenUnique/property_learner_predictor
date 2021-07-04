GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_dec_clevrer.py \
    --load_reference_flag 1 \
    --sim_data_flag 1 \
    --ann_dir ../../../output/causal_output/causal_sim_v13/motions \
    --ref_dir ../../../output/causal_output/reference_v13/motions \
    --batch_size 16 \
    --num_workers 4 \
    --dims 8 \
    --timesteps 1 \
    --epochs 100 \
    --train_st_idx 0 \
    --train_ed_idx 8000 \
    --test_st_idx 9000 \
    --test_ed_idx 10000 \
    --val_st_idx 8000 \
    --val_ed_idx 9000 \
