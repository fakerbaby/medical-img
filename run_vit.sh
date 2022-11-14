export CUDA_VISIBLE_DEVICES=7
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

debug="true"
DATE=`date +%Y%m%d`
model_name=vit_net
dataset=vit_data
TASK_TYPE=base

bsz=16
seed=42
lr=1e-4
weight_decay=1e-2
max_epochs=200
lr_scheduler="cosine"
loss="focal"
optimizer='AdamW'
report_to='wandb'
#earlystop
patience=10
precision=32
gradient_clip_val=0.1
alpha=0.25
gamma=2
ver=0
v_num=0

exp_name=${model_name}.${dataset}.${bsz}.${lr}.${loss}.${optimizer}.${alpha}.${gamma}.${DATE}.${max_epochs}.${precision}.${gradient_clip_val}.${TASK_TYPE}
SAVE=lightning_logs/${exp_name}

echo "${SAVE}"

if $debug == "true"
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
        --report_to ${report_to}\
        --no_augment 
        # --precision ${precision} \     
        # --load_ver ${exp_name} \
        # --load_v_num ${exp_name} \
else
    nohup  python core/main.py --gpus=1 \
        --gradient_clip_val ${gradient_clip_val} \
        --model_name ${model_name} \
        --dataset ${dataset} \
        --batch_size ${bsz} \
        --num_workers 0 \
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
        --no_augment \
        --report_to ${report_to}\
        >> lancet/single_gpu_${exp_name}.log 2>&1 &
fi