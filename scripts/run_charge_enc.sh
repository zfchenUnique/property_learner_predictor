GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_enc_clevrer.py \
    --load_reference_flag 1 \
    --train_st_idx 0 \
    --train_ed_idx 3000 \
    --test_st_idx 3000 \
    --test_ed_idx 4000 \
    --val_st_idx  4000 \
    --val_ed_idx 4968 \
    --dims 11 \
    --sim_data_flag 0 \
    --batch_size 4 \
    --num_workers 0 \
    --ann_dir ../../../output/render_output_disk2/causal_v14 \
    --ref_dir ../../../output/render_output_disk2/reference_v14 \
    --track_dir ../../../output/render_output_disk2/box_v14 \
    --ref_track_dir ../../../output/render_output_disk2/box_reference_v14 \
    --proposal_flag  1 \
    
    #--ann_dir ../../../output/render_output/ann_v13 \
    #--ref_dir ../../../output/render_output/reference_v13 \
    #--track_dir ../../../output/render_output/box_v13 \
    #--ref_track_dir ../../../output/render_output/box_reference_v13 \
