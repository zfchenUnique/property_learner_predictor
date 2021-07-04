GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer.py \
    --load_reference_flag 1 \
    --save-folder logs/exp2021-05-11T07:04:47.415134 \
    --sim_data_flag 1 \
    --dims 8 \
    --test_st_idx 800 \
    --test_ed_idx 810 \
    --visualize_flag 1 \
    --batch_size 4 \
    --num_workers 1 \
    --ann_dir ../../../output/causal_output/causal_sim_v13/motions \
    --ref_dir ../../../output/causal_output/reference_v13/motions \
    #--decoder rnn \
