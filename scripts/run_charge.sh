GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_clevrer.py \
    --suffix _charged5 
