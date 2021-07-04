GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_enc_clevrer.py \
    --load_reference_flag 1 \
    --train_st_idx 0 \
    --train_ed_idx 1900 \
    --test_st_idx 1900 \
    --test_ed_idx 2000 \
    --val_st_idx  1900 \
    --val_ed_idx 2000 \
    --dims 11 \
    --sim_data_flag 0 \
    --batch_size 64 \
    --ann_dir /home/zfchen/code/output/ns-vqa_output/v13_prp/config \
    --ref_dir /home/zfchen/code/output/ns-vqa_output/v13_prp_reference/config \
    --track_dir /home/zfchen/code/output/ns-vqa_output/v13_prp/tracks \
    --ref_track_dir /home/zfchen/code/output/ns-vqa_output/v13_prp_reference/tracks \
    --num_workers 0 \
