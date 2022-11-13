export CUDA_VISIBLE_DEVICES=5
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# export PYTHONHASHSEED=0

debug="true"
DATE=`date +%Y%m%d`
model_name=res_net152
dataset=resnet_data
TASK_TYPE=base

bsz=16
seed=42
lr=5e-6
weight_decay=5e-1
max_epochs=300
lr_scheduler="cosine"
loss="focal"
optimizer='AdamW'
report_to="none"

#earlystop
patience=20
# precision=16
gradient_clip_val=0.1
alpha=0.25
gamma=2

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
        --no_augment  \
        --patience ${patience}
        # --precision ${precision} \     
        # --load_ver ${exp_name} \
        # --load_v_num ${exp_name} \
else
    nohup python core/main.py --gpus=1 \
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
        --log_dir ${exp_name}\
        --lr_scheduler ${lr_scheduler} \
        --weight_decay ${weight_decay} \
        --optimizer ${optimizer} \
        --gamma ${gamma} \
        --alpha ${alpha} \
        --no_augment  \
        --patience ${patience} \
        >> ${exp_name}.log 2 >&1 &
fi