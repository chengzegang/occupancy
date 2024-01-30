export HF_HOME=/scratch/zc2309/.cache/huggingface
python -m occupancy.pipelines.train --model panoramic2voxel --data-dir /nuscenes/nuscenes \
    --cache-dir .cache/nuscenes --batch-size 8 --num-workers 16 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 64 --save-every 100 --weight-decay 1.0e-1 \
    --num-classes 1 --ema --lr 1.0e-3