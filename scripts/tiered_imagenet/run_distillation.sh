gpuid=0

#DATA_ROOT=/root/data/tiered_imagenet
DATA_ROOT=/home/okc/data/tiered_imagenet
MODEL_PATH=./checkpoints/tiered_imagenet/ResNet12_stl_deepbdc_pretrain-144-SCAttn/last_model.tar
cd ../../../

echo "============= distill born 1 ============="
python distillation.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 90 --milestones 40 70 --save_freq 100 --teacher_path $MODEL_PATH --trial 1 --reduce_dim 144 --dropout_rate 0.5 --val last