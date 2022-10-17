export CUDA_VISIBLE_DEVICES=5
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d`
model_name=baseline_net
TASK_TYPE=base

bsz=16
seed=0
lr=5e-5
lr_scheduler="step"
loss="l1"
exp_name=${model_name}.${TASK_TYPE}.${bsz}.${lr}.${loss}.${DATE}
SAVE=lightning_logs/${exp_name}
echo "${SAVE}"

python core/main.py --gpus=1 \
    --model_name ${model_name} \
    --batch_size ${bsz} \
    --num_workers 0 \
    --seed  ${seed} \
    --lr ${lr} \
    --loss ${loss} \
    --data_dir "./dataset" \
    --log_dir "lightning_logs"\
    --load_ver ${exp_name} \
    --lr_scheduler ${lr_scheduler} \
    # >> ${SAVE}/single_gpu_${version}.log 2>&1 &
