gpuid=0


DATA_ROOT=/home/okc/data/tiered_imagenet
MODEL_PATH=./checkpoints/tiered_imagenet/ResNet12_stl_deepbdc_distill_born1/last_model.tar
cd ../../../

echo "============= meta-test 1-shot ============="
#python test.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --n_shot 1 --model_path $MODEL_PATH --test_task_nums 5 --penalty_C 0.1 --reduce_dim 144 --test_n_episode 2000

echo "============= meta-test 5-shot ============="
python test.py --dataset tiered_imagenet --data_path $DATA_ROOT --model ResNet12 --method stl_deepbdc --image_size 84 --gpu ${gpuid} --n_shot 5 --model_path $MODEL_PATH --test_task_nums 5 --penalty_C 40 --reduce_dim 144 --test_n_episode 2000 --dropout_rate 0.6