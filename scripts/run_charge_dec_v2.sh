GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_dec_clevrer_v2.py \
    --load_reference_flag 1 \
    --sim_data_flag 0 \
    --ann_dir ../../../output/render_output/ann_v13 \
    --ref_dir ../../../output/render_output/reference_v13 \
    --track_dir ../../../output/render_output/box_v13 \
    --ref_track_dir ../../../output/render_output/box_reference_v13 \
    --train_st_idx 0 \
    --train_ed_idx 8000 \
    --test_st_idx 9000 \
    --test_ed_idx 10000 \
    --val_st_idx 8000 \
    --val_ed_idx 9000 \
    --batch_size 16 \
    --num_workers 4 \
    --dims 8 \
    --exclude_field_video 1 \
    --save_str no_field \
