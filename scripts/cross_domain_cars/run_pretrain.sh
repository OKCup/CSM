gpuid=0

DATA_ROOT=/home/okc/data/miniImageNet_raw/ # path to the json file of CUB
cd ../../

echo "============= pre-train ============="
python pretrain.py --dataset cross_domain_cars --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --lr 5e-2 --t_lr 1e-3 --epoch 150 --wd 5e-4 --milestones 80 120 --save_freq 100 --reduce_dim 256 --dropout_rate 0.8 --val last