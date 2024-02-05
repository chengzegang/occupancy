export WANDB_CACHE_DIR=/scratch/$USER/.cache/wandb
export WANDB_MODE=disabled
python -m occupancy.pipelines.train --model autoencoderkl --data-dir /nuscenes/nuscenes \
    --cache-dir .cache/nuscenes --batch-size 8 --num-workers 16 --device cuda \
    --dtype bf16 --model-config config/diffusion_image2voxel.yml --grad-accum 4 --lr 5.0e-5 --num-classes 18 --weight-decay 1.0e-3

