gpuid=0

#DATA_ROOT=/home/okc/data/CUB/ # path to the json file of CUB
DATA_ROOT=/home/okc/data/Cars/
MODEL_PATH=./checkpoints/cross_domain_cub/ResNet12_stl_deepbdc_distill_born3/last_model.tar
cd ../../

#echo "============= meta-test 1-shot ============="
#python test.py --dataset cub --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 224 --gpu ${gpuid} --n_shot 1 --model_path $MODEL_PATH --reduce_dim 256 --test_task_nums 5 --penalty_C 20


echo "============= meta-test 5-shot ============="
python test.py --dataset Cars --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --n_shot 5 --model_path $MODEL_PATH --reduce_dim 256 --test_task_nums 5 --penalty_C 20