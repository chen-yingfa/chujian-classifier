#!/bin/bash
#SBATCH -G 1
#SBATCH -w thunlp-215-5

lr="0.005"

pretrained="True"
model_name="vit_base_patch16_224"
output_dir="result/glyphs_955"

cmd="python3 train_vit.py"
cmd+=" --batch_size 64"
cmd+=" --num_epochs 16"
cmd+=" --lr ${lr}"
cmd+=" --output_dir ${output_dir}"
cmd+=" --log_interval 10"
cmd+=" --mode train_test"
cmd+=" --model_name ${model_name}"
cmd+=" --pretrained ${pretrained}"

$cmd
