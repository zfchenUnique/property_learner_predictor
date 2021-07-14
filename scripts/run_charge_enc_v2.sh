GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_enc_clevrer.py \
    --train_st_idx 0 \
    --train_ed_idx 10000 \
    --test_st_idx 0 \
    --test_ed_idx 5000 \
    --val_st_idx 0 \
    --val_ed_idx 4969 \
    --dims 11 \
    --sim_data_flag 0 \
    --batch_size 64 \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/farm2/causal_v15 \
    --ref_dir /home/zfchen/code/output/render_output_vislab3/reference_v15 \
    --track_dir /home/zfchen/code/output/render_output_vislab3/box_v15 \
    --ref_track_dir /home/zfchen/code/output/render_output_vislab3/box_reference_v15 \
    --ann_dir_val /home/zfchen/code/output/render_output_disk2/causal_v14 \
    --ref_dir_val /home/zfchen/code/output/render_output_disk2/reference_v14 \
    --track_dir_val /home/zfchen/code/output/render_output_disk2/box_v14 \
    --ref_track_dir_val /home/zfchen/code/output/render_output_disk2/box_reference_v14 \
    --data_noise_aug 1 \
    --data_noise_weight 0.001 \
    --num_workers 2 \
    --mask_aug_prob 0 \
    --save_str v15_mass_ref_aug_noise_001 \
    --load_reference_flag 1 \
    --mass_only_flag 1 \
    --ref_num_aug 1 \
    --charge_only_flag 0 \
    #--proposal_flag  1 \
    #--ann_dir ../../../output/render_output/ann_v13 \
    #--ref_dir ../../../output/render_output/reference_v13 \
    #--track_dir ../../../output/render_output/box_v13 \
    #--ref_track_dir ../../../output/render_output/box_reference_v13 \