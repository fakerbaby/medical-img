export CUDA_VISIBLE_DEVICES=7
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

DATE=`date +%Y%m%d`
model_name=baseline_net
dataset=img512_data
TASK_TYPE=base

bsz=16
seed=0
lr=5e-6
lr_scheduler="step"
loss="bce"
optimizer="AdamW"
exp_name=${model_name}.${dataset}.${bsz}.${lr}.${loss}.${optimizer}.${DATE}.${TASK_TYPE}
SAVE=lightning_logs/${exp_name}
echo "${SAVE}"

nohup python core/main.py --gpus=1 \
    --model_name ${model_name} \
    --batch_size ${bsz} \
    --num_workers 0 \
    --seed  ${seed} \
    --lr ${lr} \
    --loss ${loss} \
    --dataset ${dataset} \
    --data_dir "./dataset" \
    --log_dir "lightning_logs"\
    --load_ver ${exp_name} \
    --lr_scheduler ${lr_scheduler} \
    --optimizer ${optimizer} \
    >> single_gpu_${exp_name}.log 2>&1 &
