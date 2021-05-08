GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_enc_clevrer.py \
    --suffix _charged5 \
    --batch_size 4 \
    --num_workers 0 \
    --train_st_idx 0 \
    --train_ed_idx 120 \
    --load_reference_flag 1 \
