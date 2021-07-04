GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer.py \
    --batch_size 4 \
    --num_workers 1 \
    --train_st_idx 0 \
    --train_ed_idx 100 \
    --load_reference_flag 1 \
    --save-folder logs/exp2021-05-10T16:21:21.846955 \
    --sim_data_flag 0 \
    --dims 11 \
    --test_st_idx 800 \
    --test_ed_idx 810 \
    --visualize_flag 1 \
    --ann_dir ../../../output/render_output/ann_v13 \
    --ref_dir ../../../output/render_output/reference_v13 \
    --track_dir ../../../output/render_output/box_v13 \
    --ref_track_dir ../../../output/render_output/box_reference_v13 \
    --decoder rnn \
