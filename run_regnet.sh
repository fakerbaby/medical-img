export CUDA_VISIBLE_DEVICES=2
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# export PYTHONHASHSEED=0

hungup="none"
DATE=`date +%Y%m%d`
model_name=reg_net
dataset=resnet_data
TASK_TYPE=base

bsz=8
seed=42
lr=3e-6
weight_decay=5e-1
max_epochs=300
lr_scheduler="cosine"
loss="focal"
optimizer='AdamW'
report_to='wandb'
warmup_steps=5000

#earlystop
patience=50
precision=16
gradient_clip_val=0.1
alpha=0.05
gamma=2

exp_name=${model_name}.${dataset}.${bsz}.${lr}.${loss}.${weight_decay}.${optimizer}.${alpha}.${gamma}.${DATE}.${max_epochs}.${precision}.${warmup_steps}.${gradient_clip_val}.${TASK_TYPE}
SAVE=lightning_logs/${exp_name}

echo "${SAVE}"

if $hungup == "true"
then
    python core/main.py --gpus=1 \
        --gradient_clip_val ${gradient_clip_val} \
        --model_name ${model_name} \
        --dataset ${dataset} \
        --batch_size ${bsz} \
        --num_workers 8 \
        --max_epochs ${max_epochs} \
        --seed  ${seed} \
        --lr ${lr} \
        --loss ${loss} \
        --data_dir "./dataset" \
        --log_dir ${exp_name}\
        --lr_scheduler ${lr_scheduler} \
        --weight_decay ${weight_decay} \
        --optimizer ${optimizer} \
        --gamma ${gamma} \
        --alpha ${alpha} \
        --no_augment  \
        --report_to ${report_to} \
        --warmup_steps ${warmup_steps} \
        --patience ${patience}
        # --precision ${precision}   
        # --load_ver ${exp_name} \
        # --load_v_num ${exp_name} \
else
    nohup  python core/main.py --gpus=1 \
        --gradient_clip_val ${gradient_clip_val} \
        --model_name ${model_name} \
        --dataset ${dataset} \
        --batch_size ${bsz} \
        --num_workers 8 \
        --max_epochs ${max_epochs} \
        --seed  ${seed} \
        --lr ${lr} \
        --loss ${loss} \
        --data_dir "./dataset" \
        --log_dir ${exp_name}\
        --lr_scheduler ${lr_scheduler} \
        --weight_decay ${weight_decay} \
        --optimizer ${optimizer} \
        --gamma ${gamma} \
        --alpha ${alpha} \
        --no_augment  \
        --report_to ${report_to} \
        --warmup_steps ${warmup_steps} \
        --patience ${patience} \
        >> ${exp_name}.log 2>&1 &
fi