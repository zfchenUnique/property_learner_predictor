GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_enc_clevrer.py \
    --load_reference_flag 1 \
    --sim_data_flag 1 \
    --dims 8 \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16/causal_sim/motions \
    --ref_dir /home/zfchen/code/output/render_output_vislab3/v16/reference/motions \
    --train_st_idx 0 \
    --train_ed_idx 2000 \
    --test_st_idx 1900 \
    --test_ed_idx 2000 \
    --val_st_idx  1900 \
    --val_ed_idx 2000 \
    --batch_size 16 \
    --num_workers 4 \
    --decoder rnn \
