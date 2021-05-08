GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_enc_clevrer.py \
    --suffix _charged5 \
    --batch_size 4 \
    --num_workers 0 \
    --load_reference_flag 1 \
    --save-folder logs/exp107 \
    --sim_data_flag 1 \
    --dims 7 \
    --test_st_idx 4000 \
    --test_ed_idx 5000 \
    --ann_dir ../../../output/causal_output/causal_sim_v11_4/motions \
    --ref_dir ../../../output/causal_output/reference_v11_4/motions \
