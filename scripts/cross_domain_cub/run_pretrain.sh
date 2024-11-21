gpuid=0

DATA_ROOT=/home/okc/data/miniImageNet_raw/ # path to the json file of CUB
cd ../../

echo "============= pre-train ============="
python pretrain.py --dataset cross_domain_cub --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 5e-2 --epoch 180 --wd 5e-4 --milestones 100 150 --save_freq 100 --reduce_dim 256 --dropout_rate 0.6 --val last