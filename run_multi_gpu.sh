export CUDA_VISIBLE_DEVICES=4,5

version=1017
log_dir=lightning_logs/

nohup python main.py --gpus=2 \
    --load_v_num ${version} \
    >> ${log_dir}/multi_gpu.log 2>&1 &