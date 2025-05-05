#!/bin/bash

post='run'
printFreq=100


dataset='retinal_disease'

pretrainedModel='../pretrained_model/resnet101-5d3b4d8f.pth'
resumeModel='./checkpoint/run/Checkpoint_Best.pth'
evaluate=True


epochs=20
startEpoch=0
stepEpoch=10

batchSize=18
lr=1e-5
momentum=0.9
weightDecay=5e-4

workers=8

gen_psl_epoch=5

input_size=512
alpha=0.95
lam=0.6

python ../PSL_multiDisease_detec.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --dataset ${dataset} \
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --workers ${workers} \
    --gen_psl_epoch ${gen_psl_epoch} \
    --input_size ${input_size} \
    --alpha ${alpha} \
    --lam ${lam} \
