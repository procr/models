#!/bin/bash
set -x

#model=ResNet50
#model=VGG19
#model=AlexNet
#model=GoogLeNet
#model=InceptionV4
model=SE_ResNeXt50_32x4d
#model=ResNeXt50_32x4d
#model=MobileNetV1_x1_0

precision=int16
#precision=int8

batch_size=1

n_iters=10

place=xpu
#place=cpu

run_mode=fused_infer
#run_mode=infer

python eval.py \
    --batch_size=$batch_size \
    --n_iters=$n_iters \
    --place=$place \
    --run_mode=$run_mode \
    --model=$model \
    --precision=$precision \
    --data_dir=/chenrong06/ILSVRC2012/ \
    --pretrained_model=/chenrong06/pretraind_models/$model\_pretrained
