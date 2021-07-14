GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_enc_clevrer_v2.py \
    --load_reference_flag 1 \
    --test_st_idx 4000 \
    --test_ed_idx 6000 \
    --dims 11 \
    --sim_data_flag 0 \
    --num_workers 2 \
    --batch_size 2 \
    --proposal_flag 0 \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16/render/causal_sim \
    --ref_dir /home/zfchen/code/output/render_output_vislab3/v16/render/reference \
    --track_dir /home/zfchen/code/output/render_output_vislab3/v16/box \
    --ref_track_dir /home/zfchen/code/output/render_output_vislab3/v16/box_reference \
    --version v16_val \
    --mass_best_flag 1 \
    --save-folder  logs/exp_v16_mass_noise_001_render
    #--charge_best_flag 1 \
    #--save-folder logs/exp_v16_charge_noise_001_render \
    #--mass_best_flag 1 \
    #--save-folder logs/exp_v15_mass_ref_aug_noise_001 \
    #--charge_best_flag 1 \
    #--save-folder logs/exp_v15_charge_only_ref_aug_full \
    #--mass_best_flag 1 \
    #--save-folder logs/exp_v15_mass_ref_aug_noise_001 \
    #--save-folder logs/exp19 \
    #--save-folder logs/exp_v15_encode_noise_0001 \
    #--save-folder logs/exp_v15_encode_ref_1_4 \
    #--mass_best_flag 1 \
    #--save-folder logs/exp_v15_encoder \
    #--save-folder logs/exp_v15_encode_noise_0001 \
    #--mass_best_flag 1 \

    
    #--ann_dir ../../../output/render_output/ann_v13 \
    #--ref_dir ../../../output/render_output/reference_v13 \
    #--track_dir ../../../output/render_output/box_v13 \
    #--ref_track_dir ../../../output/render_output/box_reference_v13 \
    #--charge_best_flag 1 \
    #--mass_best_flag 1 \
    #--ann_dir /home/zfchen/code/output/ns-vqa_output/v13_prp/config \
    #--ref_dir /home/zfchen/code/output/ns-vqa_output/v13_prp_reference/config \
    #--track_dir /home/zfchen/code/output/ns-vqa_output/v13_prp/tracks \
    #--ref_track_dir /home/zfchen/code/output/ns-vqa_output/v13_prp_reference/tracks \
