GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer.py \
    --suffix _charged5 \
    --batch_size 4 \
    --num_workers 1 \
    --train_st_idx 0 \
    --train_ed_idx 100 \
    --load_reference_flag 1 \
    --save-folder logs/exp2021-05-05T21:12:16.343354 \
    --sim_data_flag 1 \
    --dims 7 \
    --ann_dir ../../../output/causal_output/causal_sim_v11_4/motions \
    --ref_dir ../../../output/causal_output/reference_v11_4/motions \
    --test_st_idx 4000 \
    --test_ed_idx 4100 \
    --visualize_flag 1 \
