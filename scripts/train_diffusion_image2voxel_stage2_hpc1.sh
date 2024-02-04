export HF_HOME=/scratch/zc2309/.cache/huggingface
export TORCH_HOME=/scratch/zc2309/.cache/torch
python -m occupancy.pipelines.train --model panoramic2voxel --data-dir /nuscenes/nuscenes \
    --cache-dir .cache/nuscenes --batch-size 8 --num-workers 16 --device cuda \
    --dtype fp32 --model-config config/diffusion_image2voxel.yml --grad-accum 1 --save-every 100 --weight-decay 1.0e-5 \
    --num-classes 1 --lr 1.0e-4