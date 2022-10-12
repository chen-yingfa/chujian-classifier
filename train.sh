#!/bin/bash
#SBATCH -G 1

lr="0.00005"

output_dir="result/glyphs_955_lr${lr}/"

cmd="python3 train_resnet.py"
cmd+=" --batch_size 256"
cmd+=" --num_epochs 4"
cmd+=" --lr ${lr}"
cmd+=" --output_dir ${output_dir}"
cmd+=" --mode train_test"

$cmd