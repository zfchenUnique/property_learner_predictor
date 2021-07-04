GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_enc_clevrer_v2.py \
    --load_reference_flag 1 \
    --test_st_idx 0 \
    --test_ed_idx 5000 \
    --dims 11 \
    --sim_data_flag 0 \
    --num_workers 0 \
    --batch_size 1 \
    --proposal_flag 1 \
    --ref_dir /home/zfchen/code/output/ns-vqa_output/v11_prp_reference_debug/config \
    --ref_track_dir /home/zfchen/code/output/ns-vqa_output/v11_prp_reference_debug/tracks \
    --ann_dir /home/zfchen/code/output/ns-vqa_output/v11_prp_refine_debug/config \
    --track_dir /home/zfchen/code/output/ns-vqa_output/v11_prp_refine_debug/tracks \
    --version v11_debug \
    --mass_best_flag 1 \
    --save-folder logs/exp_v15_mass_ref_aug_noise_001 \
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
