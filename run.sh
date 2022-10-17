export CUDA_VISIBLE_DEVICES=0,1

version=1017
log_dir=/core/lightning_logs/

nohup python main.py --gpus=1 \
    --batch_size 16 \
    --num_workers 8 \
    --seed  0 \
    --lr 5e-5 \
    >> ${log_dir}/single_gpu_${version}.log 2>&1 &
