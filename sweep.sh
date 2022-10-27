export CUDA_VISIBLE_DEVICES=3
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# export PYTHONHASHSEED=0

debug="true"
DATE=`date +%Y%m%d`
model_name=res_net152
dataset=resnet_data
TASK_TYPE=base

bsz=16
seed=0
lr=5e-2
weight_decay=1e-1
max_epochs=1
lr_scheduler="step"
loss="focal"
optimizer='AdamW'

#earlystop
patience=20
# precision=16
gradient_clip_val=0.1
alpha=0.25
gamma=2

exp_name=sweep
SAVE=wandb/${exp_name}

echo "${SAVE}"

if $debug == "true"
then
    python core/sweep.py --gpus=1 \
        --gradient_clip_val ${gradient_clip_val} \
        --model_name ${model_name} \
        --dataset ${dataset} \
        --batch_size ${bsz} \
        --num_workers 16 \
        --max_epochs ${max_epochs} \
        --seed  ${seed} \
        --lr ${lr} \
        --loss ${loss} \
        --data_dir "./dataset" \
        --log_dir ${SAVE}\
        --lr_scheduler ${lr_scheduler} \
        --weight_decay ${weight_decay} \
        --optimizer ${optimizer} \
        --gamma ${gamma} \
        --alpha ${alpha} \
        --no_augment  \
        --patience ${patience}
        # --precision ${precision} \     
        # --load_ver ${exp_name} \
        # --load_v_num ${exp_name} \
else
    nohup python core/main.py --gpus=1 \
        --model_name ${model_name} \
        --dataset ${dataset} \
        --batch_size ${bsz} \
        --num_workers 8 \
        --max_epochs ${max_epochs} \
        --seed  ${seed} \
        --lr ${lr} \
        --loss ${loss} \
        --data_dir "./dataset" \
        --log_dir "lightning_logs"\
        --load_ver ${exp_name} \
        --load_v_num ${exp_name} \
        --lr_scheduler ${lr_scheduler} \
        --weight_decay ${weight_decay} \
        --optimizer ${optimizer} \
        --gamma ${gamma} \
        --alpha ${alpha} \
        >> single_gpu_${exp_name}.log 2>&1 &
fi