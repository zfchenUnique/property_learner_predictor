GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_dec_clevrer_v2.py \
    --batch_size 4 \
    --num_workers 1 \
    --load_reference_flag 1 \
    --test_st_idx 8000 \
    --test_ed_idx 9000 \
    --visualize_flag 1 \
    --ann_dir ../../../output/render_output/ann_v13 \
    --ref_dir ../../../output/render_output/reference_v13 \
    --track_dir ../../../output/render_output/box_v13 \
    --ref_track_dir ../../../output/render_output/box_reference_v13 \
    --dims 8 \
    --sim_data_flag 0 \
    --save-folder logs/exp_no_field \
    --exclude_field_video 1 \
    --vis_dir ../../../output/dynamics
    #--save-folder logs/exp2021-05-11T16:42:32.009489 \
