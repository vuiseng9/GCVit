#!/usr/bin/env bash

export WANDB_DISABLED=false # Enable wandb
export WANDB_WATCH=false # Disable gradient serialization to wandb
export WANDB_USERNAME=vchua
export WANDB_API_KEY=f8a95080288950342f1695008cd8256adc3b0778

#------------------------------------------------------
export WANDB_PROJECT="(hgx1) gcvit"
export CUDA_VISIBLE_DEVICES=0,1,2,3

RUNID=gcvit_base_img1k
TRAINCFG=/data/vchua/dev/msft-swin/gcvit/configs/gc_vit_base.yml

OUTROOT=/data/vchua/run/msft-swin
WORKDIR=/data/vchua/dev/msft-swin/gcvit

CONDAROOT=/data/vchua/miniconda3/
CONDAENV=msft-swin

OUTDIR=$OUTROOT/$RUNID
cd $WORKDIR
mkdir -p $OUTDIR

cmd="
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    --master_port 12345 \
    train.py \
    --config $TRAINCFG \
    --data_dir /data/dataset/imagenet/ilsvrc2012/torchvision/ \
    --batch-size 32 \
    --native-amp \
    --model-ema \
    --output $OUTDIR \
    --log-wandb \
    --tag $RUNID
"

# python -m torch.distributed.launch --nproc_per_node <num-of-gpus> --master_port 11223  train.py \ 
# --config <config-file> --data_dir <imagenet-path> --batch-size <batch-size-per-gpu> --tag <run-tag> --model-ema

source $CONDAROOT/etc/profile.d/conda.sh
conda activate ${CONDAENV}
eval $cmd 