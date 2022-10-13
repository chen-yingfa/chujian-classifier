#!/bin/bash
#SBATCH -G 1

lr="0.001"

model_name="resnet50"
output_dir="result/glyphs_955_${model_name}_lr${lr}/"

cmd="python3 train_resnet.py"
cmd+=" --batch_size 512"
cmd+=" --num_epochs 4"
cmd+=" --lr ${lr}"
cmd+=" --output_dir ${output_dir}"
cmd+=" --mode train_test"
cmd+=" --model_name ${model_name}"
cmd+=" --pretrained False"

$cmd