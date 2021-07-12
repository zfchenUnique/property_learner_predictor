GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_enc_clevrer.py \
    --load_reference_flag 1 \
    --sim_data_flag 1 \
    --dims 8 \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16/causal_sim/motions \
    --ref_dir /home/zfchen/code/output/render_output_vislab3/v16/reference/motions \
    --ann_dir_val /home/zfchen/code/output/render_output_vislab3/v16/causal_sim/motions \
    --ref_dir_val /home/zfchen/code/output/render_output_vislab3/v16/reference/motions \
    --train_st_idx 0 \
    --train_ed_idx 4000 \
    --test_st_idx 4000 \
    --test_ed_idx 5000 \
    --val_st_idx  4000 \
    --val_ed_idx 5000 \
    --batch_size 16 \
    --data_noise_aug 1 \
    --data_noise_weight 0.001 \
    --num_workers 2 \
    --mask_aug_prob 0 \
    --load_reference_flag 0 \
    --mass_only_flag 1 \
    --ref_num_aug 0 \
    --charge_only_flag 0 \
    --save_str v16_mass_only_noise_001_sim_no_ref_aug \
    #--save_str v15_mass_ref_aug_noise_001_sim \
