#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

budget_flops=20000e6
max_layers=60
input_image_size=640
population_size=512
evolution_max_iter=480000  # we suggest evolution_max_iter=480000 for
# evolution_max_iter=40000

save_dir=./save_dir/Zen_yolov4_flops20G
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperResK3K3(8,16,1,8,1)SuperResK3K3(16,32,2,16,1)SuperResK3K3(32,64,2,32,1)SuperResK3K3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

export CUDA_VISIBLE_DEVICES=0

python evolution_search.py --gpu 0 \
  --zero_shot_score Zen \
  --search_space SearchSpace/search_space_XXBL.py \
  --budget_flops ${budget_flops} \
  --max_layers ${max_layers} \
  --batch_size 64 \
  --input_image_size ${input_image_size} \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 100 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir} \
  --yaml_file  models/yolov4-p5_test.yaml

python analyze_model.py --img 640 --cfg ${save_dir}/best_structure.yaml --save_dir ${save_dir}
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 32 --img ${input_image_size} ${input_image_size} --data coco.yaml --cfg ${save_dir}/best_structure.yaml --weights '' --hyp 'data/hyp.scratch.yaml' --sync-bn --device 0,1,2,3 --name yolov4-p5 --epochs 300 --resume
# python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 32 --img 640 640 --data coco.yaml --cfg yolov4-p5_test.yaml --weights '' --sync-bn --device 2,3 --name yolov4-p5

# python test.py --img 640 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p5.pt
