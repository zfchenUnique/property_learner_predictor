GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_dec_clevrer.py \
    --suffix _charged5 \
    --load_reference_flag 1 \
    --sim_data_flag 1 \
    --ann_dir ../../../output/causal_output/causal_sim_v11_4/motions \
    --ref_dir ../../../output/causal_output/reference_v11_4/motions \
    --batch_size 16 \
    --num_workers 4 \
    --dim 7 \
    --timesteps 125 \
    --epochs 1 \
    --train_st_idx 0 \
    --train_ed_idx 200 \
    --test_st_idx 300 \
    --test_ed_idx 400 \
    --val_st_idx 200 \
    --val_ed_idx 300 \
